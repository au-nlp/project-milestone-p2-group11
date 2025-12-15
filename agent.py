import json
import torch
import time
import os
import csv
import functools
import logging
import numpy as np
import torch.nn.functional as F
from groq import Groq
from groq.types.chat.chat_completion_message import ChatCompletionMessage
from openai import OpenAI
from tqdm import tqdm
import networkx as nx
import pandas as pd

from prompt import Prompt
from config_local import Config
from semantic import SemanticAnalyser


logger = logging.getLogger(__name__)


PRICE_MAPPER = {
    'openai-gpt-oss-20b': {
        'input': 0.05,
        'output': 0.45,
    },
    'openai-gpt-oss-120b': {
        'input': 0.1,
        'output': 0.7,
    }
}


def digitalocean_retry(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        delays = [30, 60, 120]
        for attempt, delay in enumerate(delays, start=1):
            print(f'Attempt {attempt} starting.')
            try:
                result = func(*args, **kwargs)
                print(f'Succeeded on attempt {attempt}.')
                return result
            except Exception as e:
                print(f'Attempt {attempt} failed: {e}.\n')

                # If we've exhausted retries, re-raise
                if attempt == len(delays):
                    raise RuntimeError(
                        f'Failed after {attempt} retries.'
                    ) from e

                print(f'Retrying in {delay}s (next attempt {attempt+1}/{len(delays)})')
                time.sleep(delay)
    return wrapper


class Agent:
    def __init__(self, config: Config, link_mappings) -> None:
        self.link_mappings = link_mappings

        self.llm_config = config.llm_config
        self.base_branch = config.base_branch
        self.max_depth = config.max_depth
        self.debug = config.debug
        self.results_folder = config.results_folder

        if self.debug:
            self.client = Groq(api_key=config.groq_api_key)
        else:
            self.client = OpenAI(
                base_url=config.digital_ocean_url,
                api_key=config.digital_ocean_api_key,
            )

        self.generate_func = self.client.chat.completions.create
        self.prompt = Prompt(config=config)
        self.semantic_analyser = SemanticAnalyser(config=config)

    @digitalocean_retry
    def _ask_llm(self,prompt: str) -> tuple[ChatCompletionMessage, int, int]:
        result = self.generate_func(
            **self.prompt.get_config(debug=self.debug), messages=[{'role': 'user', 'content': prompt}]
        )
        choices = result.choices[0].message
        return choices, result.usage.prompt_tokens, result.usage.completion_tokens

    @digitalocean_retry
    def _ask_llm_blind(self, prompt: str) -> tuple[ChatCompletionMessage, int, int]:
        result = self.generate_func(
            **self.prompt.get_config(debug=self.debug, is_blind=True), messages=[{'role': 'user', 'content': prompt}]
        )
        choices = result.choices[0].message
        return choices, result.usage.prompt_tokens, result.usage.completion_tokens
        
    def _parse_response(
            self,
            text: ChatCompletionMessage
    ) -> list[tuple[str, int]] | None:
        content = text.content or '{}'

        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError:
            logger.error(f'LLM response is not a valid JSON: {content}')
            return None

        return [
            (guess['next_page'], guess['rating']) for guess in parsed_json['candidates']
        ]
        
    def _generate_and_score(
            self,
            current: str,
            goal: str,
            valid_links: list[str],
            history: list[str] | None = None
    ) -> tuple[list[tuple[str, int]] | None, int, int]:
        prompt = self.prompt.generate_prompt(current, goal, valid_links, memory=history)
        text, input_tokens, output_tokens = self._ask_llm(prompt)
        return self._parse_response(text), input_tokens, output_tokens

    def _generate_blind(self, start: str, goal: str) -> tuple[list[str], int, int]:
        prompt = self.prompt.generate_prompt_blind(start, goal)
        text, input_tokens, output_tokens = self._ask_llm_blind(prompt)
        content = text.content or '{}'
        parsed_json = json.loads(content)
        return parsed_json['pages'], input_tokens, output_tokens

    def _generate_and_score_with_memory(
            self, current: str, goal: str, valid_links: list[str], history: list[str]
    ) -> tuple[list[tuple[str, int]] | None, int, int]:
        prompt = self.prompt.generate_prompt(current, goal, valid_links, memory=history)
        text, input_tokens, output_tokens = self._ask_llm(prompt)
        return self._parse_response(text), input_tokens, output_tokens

    def _generate_and_score_with_external_info(
            self, current: str, goal: str, valid_links: list[str], external_info: str, history: list[str]| None = None
    ) -> tuple[list[tuple[str, int]] | None, int, int]:
        prompt = self.prompt.generate_prompt_with_external_knowledge(current, goal, valid_links, external_info, memory=history)
        text, input_tokens, output_tokens = self._ask_llm(prompt)
        return self._parse_response(text), input_tokens, output_tokens

    def navigate_tot(self, start: str, goal: str, with_memory:bool = False) -> tuple[list[str] | None, int, int]:
        incomplete_paths = [[start]]
        best_path, best_score = None, -1

        total_input_tokens = total_output_tokens = 0

        for _ in range(self.max_depth):
            new_incomplete_paths = []
            for path in incomplete_paths:
                current = path[-1]

                if current == goal:
                    return path, total_input_tokens, total_output_tokens

                links = self.link_mappings[current]
                if not links:
                    print(f'[navigate_tot] No links found for {current}')
                    continue
                if with_memory:
                    memory = path[:-1] if len(path) > 1 else []
                    llm_guesses, input_tokens, output_tokens = self._generate_and_score(current, goal, links, memory)
                else:
                    llm_guesses, input_tokens, output_tokens = self._generate_and_score(current, goal, links)

                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

                # if LLM gives no valid moves, skip this branch
                if not llm_guesses:
                    print(f'[navigate_tot] LLM returned no guesses for {current} → skipping path {path}')
                    continue

                scores = torch.tensor([float(r) for _, r in llm_guesses])
                probs = F.softmax(scores, dim=0).detach().cpu().numpy()
                probs /= probs.sum()
                branch = min(self.base_branch, len(llm_guesses))
                chosen = np.random.choice(len(llm_guesses), size=branch, p=probs, replace=False)

                for idx in chosen:
                    nxt = llm_guesses[idx][0]
                    if nxt not in path:
                        new_incomplete_paths.append(path + [nxt])

                        # track highest score
                        local_score = llm_guesses[idx][1] - 0.5 * len(path)
                        if local_score > best_score:
                            best_path, best_score = path + [nxt], local_score

            if not new_incomplete_paths:
                break
            incomplete_paths = new_incomplete_paths

        if best_path is None:
            print(f'[navigate_tot] No valid path found from {start} to {goal}. Returning None.')
        return best_path, total_input_tokens, total_output_tokens

    def navigate_link_aware(self, start: str, goal: str, with_memory:bool = False) -> tuple[list[str] | None, int, int]:
        visited = set([start])
        path = [start]
        best_path, best_score = None, -1

        total_input_tokens = total_output_tokens = 0

        for _ in range(self.max_depth):
            current = path[-1]

            if current == goal:
                return path, total_input_tokens, total_output_tokens

            links = self.link_mappings[current]
            if not links:
                continue
            if with_memory:
                memory = path[:-1] if len(path) > 1 else []
                llm_guesses, input_tokens, output_tokens = self._generate_and_score(current, goal, links, memory)
            else:
                llm_guesses, input_tokens, output_tokens = self._generate_and_score(current, goal, links)

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            # if LLM gives no valid moves, end navigation
            if not llm_guesses:
                continue
            scores = torch.tensor([float(r) for _, r in llm_guesses])
            probs = F.softmax(scores, dim=0).detach().cpu().numpy()
            probs /= probs.sum()
            #select the top n choice
            branch = min(self.base_branch, len(llm_guesses))
            chosen = np.random.choice(len(llm_guesses), size=branch, p=probs, replace=False)
            unvisited_found = False
            for idx in chosen:
                nxt = llm_guesses[idx][0]
                # only take the first unvisited link
                if nxt not in visited:
                    unvisited_found = True
                    visited.add(nxt)
                    path.append(nxt)
                    # track highest score
                    local_score = llm_guesses[idx][1] - 0.5 * len(path)
                    if local_score > best_score:
                        best_path, best_score = path.copy(), local_score
                    break
            if not unvisited_found:
                break
        result = path if path[-1] == goal else best_path
        return result, total_input_tokens, total_output_tokens

    def navigate_blind(self, start: str, goal: str) -> tuple[list[str] | None, int, int]:
        return self._generate_blind(start, goal)

    def navigate_link_aware_with_memory(self, start: str, goal: str) -> tuple[list[str] | None, int, int]:
        visited = set([start])
        path = [start]
        best_path, best_score = None, -1

        total_input_tokens = total_output_tokens = 0

        for _ in range(self.max_depth):
            current = path[-1]

            if current == goal:
                return path, total_input_tokens, total_output_tokens

            links = self.link_mappings[current]
            if not links:
                continue

            #memory is the current path
            memory = path[:-1] if len(path) > 1 else []
            llm_guesses, input_tokens, output_tokens = self._generate_and_score_with_memory(current, goal, links, memory)

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            # if LLM gives no valid moves, end navigation
            if not llm_guesses:
                continue
            scores = torch.tensor([float(r) for _, r in llm_guesses])
            probs = F.softmax(scores, dim=0).detach().cpu().numpy()
            probs /= probs.sum()
            #select the top n choice
            branch = min(self.base_branch, len(llm_guesses))
            chosen = np.random.choice(len(llm_guesses), size=branch, p=probs, replace=False)
            unvisited_found = False
            for idx in chosen:
                nxt = llm_guesses[idx][0]
                # only take the first unvisited link
                if nxt not in visited:
                    unvisited_found = True
                    visited.add(nxt)
                    path.append(nxt)
                    # track highest score
                    local_score = llm_guesses[idx][1] - 0.5 * len(path)
                    if local_score > best_score:
                        best_path, best_score = path.copy(), local_score
                    break
            if not unvisited_found:
                break
        result = path if path[-1] == goal else best_path
        return result, total_input_tokens, total_output_tokens 

    def build_link_graph(self):
        G = nx.DiGraph()
        for src, targets in self.link_mappings.items():
            for tgt in targets:
                G.add_edge(src, tgt)
        return G

    def retrieve_graph_and_semantic_info(self, goal: str, valid_links: list[str]) -> str:
        KG = self.build_link_graph()
        max_degree = max(dict(KG.degree).values()) # type: ignore
        pagerank = nx.pagerank(KG)
        pagerank_goal = pagerank.get(goal, 0)
        info = {}
        for link in valid_links:
            similarity = self.semantic_analyser.get_cosine_similarity(link, goal)
            centrality_score = abs(pagerank.get(link, 0) - pagerank_goal)
            degree = KG.degree(link) / max_degree if max_degree > 0 else 0 # type: ignore
            combined_score = 0.55 * similarity + 0.3 * centrality_score + 0.15 * degree
            info[link] = {'similarity_to_gaol': similarity, 'centrality_diff_to_goal': centrality_score, 'degree': degree, 'score': combined_score}
        links = sorted(info.items(), key=lambda x: x[1]['score'], reverse=True)[:20]
        result = {k: {kk: vv for kk, vv in v.items() if kk != "score"} for k, v in links}
        return str(result)

    def navigate_with_external_info(self, start: str, goal: str, with_memory: bool = False) -> tuple[list[str] | None, int, int]:
        visited = set([start])
        path = [start]
        best_path, best_score = None, -1

        total_input_tokens = total_output_tokens = 0

        for _ in range(self.max_depth):
            current = path[-1]

            if current == goal:
                return path, total_input_tokens, total_output_tokens

            links = self.link_mappings[current]
            if not links:
                continue

            external_info = self.retrieve_graph_and_semantic_info(goal, links)
            if with_memory:
                memory = path[:-1] if len(path) > 1 else []
                llm_guesses, input_tokens, output_tokens = self._generate_and_score_with_external_info(current, goal, links, external_info, memory)
            else:
                llm_guesses, input_tokens, output_tokens = self._generate_and_score_with_external_info(current, goal, links, external_info)

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            # if LLM gives no valid moves, end navigation
            if not llm_guesses:
                continue
            scores = torch.tensor([float(r) for _, r in llm_guesses])
            probs = F.softmax(scores, dim=0).detach().cpu().numpy()
            probs /= probs.sum()
            #select the top n choice
            branch = min(self.base_branch, len(llm_guesses))
            chosen = np.random.choice(len(llm_guesses), size=branch, p=probs, replace=False)
            unvisited_found = False
            for idx in chosen:
                nxt = llm_guesses[idx][0]
                # only take the first unvisited link
                if nxt not in visited:
                    unvisited_found = True
                    visited.add(nxt)
                    path.append(nxt)
                    # track highest score
                    local_score = llm_guesses[idx][1] - 0.5 * len(path)
                    if local_score > best_score:
                        best_path, best_score = path.copy(), local_score
                    break
            if not unvisited_found:
                break
        result = path if path[-1] == goal else best_path
        return result, total_input_tokens, total_output_tokens

    def generate_llm_paths(self, df: pd.DataFrame):
        def safe_call(func, *args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(e)
                return None

        os.makedirs(self.results_folder, exist_ok=True)
        out_path = os.path.join(self.results_folder, 'llm_paths_oss_120b.csv')
        write_header = not os.path.exists(out_path)

        # trying to recover from previous generation
        processed_ids = set()
        total_output = total_input = 0
        if not write_header:
            try:
                existing = pd.read_csv(out_path)
                processed_ids = set(existing['id'])
                print(f'Found {len(processed_ids)} already completed pairs.')

                total_input += existing[
                    ['blind_input_tokens', 'link_aware_input_tokens', 'external_info_input_tokens', 'tot_input_tokens']
                ].sum().sum()
                total_output += existing[
                    ['blind_output_tokens', 'link_aware_output_tokens', 'external_info_output_tokens', 'tot_output_tokens']
                ].sum().sum()

                self.log_cost(total_input, total_output)

            except Exception as e:
                print(f'Could not read existing file: {e}')
        
        # appending to the result file
        with open(out_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(
                    [
                        'id', 'start', 'destination', 'blind_paths', 'blind_input_tokens', 'blind_output_tokens',
                        'link_aware_paths', 'link_aware_input_tokens', 'link_aware_output_tokens',
                        'external_info_paths', 'external_info_input_tokens', 'external_info_output_tokens',
                        'tot_paths', 'tot_input_tokens', 'tot_output_tokens',
                    ]
                )

            for _, row in tqdm(df.iterrows(), total=len(df)):  # main generation logic
                id = row['id']
                if id in processed_ids:  # skip already generated id
                    print(f'Skipping {id}, since it has already been generated.')
                    continue

                start_now = time.time()

                start = row['start']
                goal = row['destination']
                rep_idx = row['replicate_idx']

                print(f'start: {start}, destination: {goal} (replicate: {rep_idx})')

                path_blind = safe_call(self.navigate_blind, start, goal)
                path_link_aware = safe_call(self.navigate_link_aware, start, goal)
                path_with_external_info = safe_call(
                    self.navigate_with_external_info, start, goal
                )
                path_tot = safe_call(self.navigate_tot, start, goal)

                def serialize(result):
                    if result is None:
                        return '', 0, 0
                    p, tin, tout = result
                    p_str = json.dumps(p if p is not None else [])  # store as JSON string
                    return p_str, tin, tout

                b_p, b_in, b_out = serialize(path_blind)
                la_p, la_in, la_out = serialize(path_link_aware)
                ei_p, ei_in, ei_out = serialize(path_with_external_info)
                tot_p, tot_in, tot_out = serialize(path_tot)

                total_output += b_out + la_out + ei_out + tot_out
                total_input += b_in + la_in + ei_in + tot_in
                print(
                    f'total input: {total_input} tokens so far\n'
                    f'total output: {total_output} tokens so far'
                )
                self.log_cost(total_input, total_output)
                end_time = time.time()

                elapsed_time = end_time - start_now
                print(f'elapsed time: {elapsed_time:.2f} seconds')

                writer.writerow(
                    [
                        id, start, goal, b_p, b_in, b_out, la_p, la_in,
                        la_out, ei_p, ei_in, ei_out, tot_p, tot_in, tot_out,
                    ]
                )

    def log_cost(self, total_input_tokens, total_output_tokens: int):
        price_info = PRICE_MAPPER.get(self.llm_config.model, {})
        input_price = price_info['input'] * total_input_tokens * 1e-6
        output_price = price_info['output'] * total_output_tokens * 1e-6
        print(f'So far input price: {input_price:.2f}$, output price: {output_price:.2f}$')

