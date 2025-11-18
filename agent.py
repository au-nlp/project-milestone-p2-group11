import json
import torch
import numpy as np
import torch.nn.functional as F
from groq import Groq
from groq.types.chat.chat_completion_message import ChatCompletionMessage
from dataclasses import asdict
import networkx as nx
from networkx.algorithms.tree import branching_weight
from pygments.lexer import combined

from prompt import Prompt
from config_local import Config
from semantic import SemanticAnalyser

class Agent:
    def __init__(self, config: Config, link_mappings) -> None:
        self.link_mappings = link_mappings

        self.llm_config = config.llm_config
        self.base_branch = config.base_branch
        self.max_depth = config.max_depth
        self.api_key = config.groq_api_key

        self.client = self._create_client()
        self.prompt = Prompt(config=config)
        self.semantic_analyser = SemanticAnalyser(config=config)

    def _create_client(self) -> Groq:
        return Groq(api_key=self.api_key)

    
    def _ask_llm(self, prompt: str) -> ChatCompletionMessage:
        return self.client.chat.completions.create(
            **self.prompt.get_config(), messages=[{'role': 'user', 'content': prompt}]
        ).choices[0].message

    def _ask_llm_blind(self, prompt: str) -> ChatCompletionMessage:
        return self.client.chat.completions.create(
            **self.prompt.get_config_blind(), messages=[{'role': 'user', 'content': prompt}]
        ).choices[0].message
        
    def _parse_response(self, text: ChatCompletionMessage) -> list[tuple[str, int]]:
        content = text.content or '{}'
        parsed_json = json.loads(content)
        return [
            (guess['next_page'], guess['rating']) for guess in parsed_json['candidates']
        ]
        
    def _generate_and_score(self, current: str, goal: str, valid_links: list[str]) -> list[tuple[str, int]]:
        prompt = self.prompt.generate_prompt(current, goal, valid_links)
        text = self._ask_llm(prompt)
        return self._parse_response(text)

    def _generate_blind(self, start: str, goal: str) -> list[str]:
        prompt = self.prompt.generate_prompt_blind(start, goal)
        text = self._ask_llm_blind(prompt)
        content = text.content or '{}'
        parsed_json = json.loads(content)
        return parsed_json['pages']

    def _generate_and_score_with_memory(self, current: str, goal: str, valid_links: list[str], history: list[str]) -> list[tuple[str, int]]:
        prompt = self.prompt.generate_prompt_with_memory(history, current, goal, valid_links)
        text = self._ask_llm(prompt)
        return self._parse_response(text)

    def _generate_and_score_with_external_info(self, current: str, goal: str, valid_links: list[str], external_info: str) -> list[tuple[str, int]]:
        prompt = self.prompt.generate_prompt_with_external_knowledge(current, goal, valid_links, external_info)
        text = self._ask_llm(prompt)
        return self._parse_response(text)

    def navigate_tot(self, start: str, goal: str) -> list[str] | None:
        visited = set([start])
        incomplete_paths = [[start]]
        best_path, best_score = None, -1

        for _ in range(self.max_depth):
            new_incomplete_paths = []
            for path in incomplete_paths:
                current = path[-1]

                if current == goal:
                    return path

                links = self.link_mappings[current]
                if not links:
                    continue

                llm_guesses = self._generate_and_score(current, goal, links)

                # if LLM gives no valid moves, skip this branch
                if not llm_guesses:
                    continue

                scores = torch.tensor([float(r) for _, r in llm_guesses])
                probs = F.softmax(scores, dim=0).detach().cpu().numpy()
                probs /= probs.sum()
                branch = min(self.base_branch, len(llm_guesses))
                chosen = np.random.choice(len(llm_guesses), size=branch, p=probs, replace=False)

                for idx in chosen:
                    nxt = llm_guesses[idx][0]
                    if nxt not in visited:
                        visited.add(nxt)
                        new_incomplete_paths.append(path + [nxt])

                        # track highest score
                        local_score = llm_guesses[idx][1] - 0.5 * len(path)
                        if local_score > best_score:
                            best_path, best_score = path + [nxt], local_score

            if not new_incomplete_paths:
                break
            incomplete_paths = new_incomplete_paths

        return best_path

    def navigate_link_aware(self, start: str, goal: str) -> list[str] | None:
        visited = set([start])
        path = [start]
        best_path, best_score = None, -1
        for _ in range(self.max_depth):
            current = path[-1]

            if current == goal:
                return path

            links = self.link_mappings[current]
            if not links:
                continue

            llm_guesses = self._generate_and_score(current, goal, links)

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
        return path if path[-1] == goal else best_path

    def navigate_blind(self, start: str, goal: str) -> list[str] | None:
        path = self._generate_blind(start, goal)
        return path

    def navigate_link_aware_with_memory(self, start: str, goal: str) -> list[str] | None:
        visited = set([start])
        path = [start]
        best_path, best_score = None, -1
        for _ in range(self.max_depth):
            current = path[-1]

            if current == goal:
                return path

            links = self.link_mappings[current]
            if not links:
                continue

            #memory is the current path
            memory = path[:-1] if len(path) > 1 else []
            llm_guesses = self._generate_and_score_with_memory(current, goal, links, memory)
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
        return path if path[-1] == goal else best_path

    def build_link_graph(self):
        G = nx.DiGraph()
        for src, targets in self.link_mappings.items():
            for tgt in targets:
                G.add_edge(src, tgt)
        return G

    def retrieve_graph_and_semantic_info(self, goal: str, valid_links: list[str]) -> list[tuple[str, int]]:
        KG = self.build_link_graph()
        max_degree = max(dict(KG.degree).values())
        pagerank = nx.pagerank(KG)
        pagerank_goal = pagerank.get(goal, 0)
        info = {}
        for link in valid_links:
            similarity = self.semantic_analyser.get_cosine_similarity(link, goal)
            centrality_score = abs(pagerank.get(link, 0) - pagerank_goal)
            degree = KG.degree(link) / max_degree if max_degree > 0 else 0
            combined_score = 0.55 * similarity + 0.3 * centrality_score + 0.15 * degree
            info[link] = {'similarity_to_gaol': similarity, 'centrality_diff_to_goal': centrality_score, 'degree': degree, 'score': combined_score}
        links = sorted(info.items(), key=lambda x: x[1]['score'], reverse=True)[:20]
        result = {k: {kk: vv for kk, vv in v.items() if kk != "score"} for k, v in links}
        return str(result)

    def navigate_with_external_info(self, start: str, goal: str) -> list[str] | None:
        visited = set([start])
        path = [start]
        best_path, best_score = None, -1
        for _ in range(self.max_depth):
            current = path[-1]

            if current == goal:
                return path

            links = self.link_mappings[current]
            if not links:
                continue

            external_info = self.retrieve_graph_and_semantic_info(goal, links)
            llm_guesses = self._generate_and_score_with_external_info(current, goal, links, external_info)

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
        return path if path[-1] == goal else best_path



