import json
import torch
import logging
import numpy as np
import torch.nn.functional as F
from groq import Groq
from groq.types.chat.chat_completion_message import ChatCompletionMessage
from openai import OpenAI

from prompt import Prompt
from config_local import Config


logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: Config, link_mappings) -> None:
        self.link_mappings = link_mappings

        self.llm_config = config.llm_config
        self.base_branch = config.base_branch
        self.max_depth = config.max_depth
        self.debug = config.debug

        if self.debug:
            self.client = Groq(api_key=config.groq_api_key)
        else:
            self.client = OpenAI(
                base_url=config.digital_ocean_url,
                api_key=config.digital_ocean_api_key,
            )

        self.generate_func = self.client.chat.completions.create
        self.prompt = Prompt(config=config)
    
    def _ask_llm(self, prompt: str) -> ChatCompletionMessage:
        return self.generate_func(
            **self.prompt.get_config(self.debug), messages=[{'role': 'user', 'content': prompt}]
        ).choices[0].message
        
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
            valid_links: list[str]
    ) -> list[tuple[str, int]] | None:
        prompt = self.prompt.generate_prompt(current, goal, valid_links)
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
