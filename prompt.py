import json
from typing import Any
from pydantic import BaseModel, Field, conlist


class Prompt:
    def __init__(self, config) -> None:
        self.config = config

    def _get_response_json_schema(self) -> dict[str, Any]:
        class Candidate(BaseModel):
            next_page: str
            # reason: str  # maybe we can do some analysis based on these reasoning later
            rating: int = Field(..., gt=0, le=10)

        class NavigationStep(BaseModel):
            candidates: conlist(
                Candidate, 
                min_length=3, 
                max_length=self.config.max_number_of_guesses_by_llm
            ) # type: ignore
        return NavigationStep.model_json_schema()

    def get_config(self, debug: bool, is_blind: bool = False) -> dict:
        schema = self._get_response_json_schema_blind() if is_blind else self._get_response_json_schema()
        base_config = {
            'model': self.config.llm_config.model,
            'max_completion_tokens': self.config.llm_config.max_completion_tokens,
            'reasoning_effort': self.config.llm_config.reasoning_effort,
            'top_p': self.config.llm_config.top_p,
            'temperature': self.config.llm_config.temperature,
            'stream': self.config.llm_config.stream,
            'stop': self.config.llm_config.stop,
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'navigation_step',
                    'schema': schema
                }
            },

        }

        if not debug:
            # force json response from digitalocean client
            json_force_config = {
                'tools': [{
                    'type': 'function',
                    'function': {
                        'name': 'navigation_step',
                        'description': 'Produces structured navigation candidates.',
                        'parameters': schema
                    }
                }],
                'tool_choice': {
                    'type': 'function',
                    'function': {'name': 'navigation_step'}
                }
            }
            base_config.update(json_force_config)
        return base_config

    def generate_prompt(self, current: str, goal: str, valid_links: list[str], memory: list[str] | None = None) -> str:
        memory_block = ""
        avoid_block = ""
        if memory:
            memory_block = f"Previous pages visited: {', '.join(memory)}\n\n"
            avoid_block = "- Do not suggest any pages that have already been visited in this path.\n\n"
        return f"""
        You are playing the Wikispeedia game.
        Current page: "{current}"
        Goal page: "{goal}"
        {memory_block}
        Think step-by-step and suggest exactly 3 possible next Wikipedia pages that could help reach the goal.
        - Only choose from this list of valid outgoing links: {valid_links}
        - For each suggestion, include:
          * next_page (must be exactly one of the valid_links)
          * rating (1–10, higher = more promising)
        {avoid_block}
        Return your answer strictly following this JSON schema:
        {self._get_response_json_schema()}
        """

    def _get_response_json_schema_blind(self) -> dict[str, Any]:
        class Path(BaseModel):
            pages: list[str]
        return Path.model_json_schema()

    def generate_prompt_blind(self, start: str, goal: str) -> str:
        return f"""
        You are playing the Wikispeedia game.
        Start page: "{start}"
        Goal page: "{goal}"
    
        Think step-by-step and suggest a possible path of Wikipedia pages that could help reach the goal.
        - Each page in the path must be a valid Wikipedia page, and there must be a link between each consecutive page in wikipedia.
        - At each step, explain briefly why you’re choosing that link.
        - The path should start with the start page and end with the goal page.
        
        Once you’ve reached the destination, write the full path.
        Return your answer strictly following this JSON schema:
        {self._get_response_json_schema_blind()}
        """

    def generate_prompt_with_memory(self, history: list[str], current: str, goal: str, valid_links: list[str]) -> str:
        return f"""
        You are playing the Wikispeedia game.
        Current page: "{current}"
        Goal page: "{goal}"
        Previous pages visited: {", ".join(history)}
    
        Think step-by-step and suggest exactly 3 possible next Wikipedia pages that could help reach the goal.
        - Only choose from this list of valid outgoing links: {valid_links}
        - For each suggestion, include:
          * next_page (must be exactly one of the valid_links)
          * rating (1–10, higher = more promising)
        - Do not suggest any pages that have already been visited in this path.
        Return your answer strictly following this JSON schema:
        {self._get_response_json_schema()}
        """

    def generate_prompt_with_external_knowledge(
        self,
        current: str,
        goal: str,
        valid_links: list[str],
        external_knowledge: str,
        memory: list[str] | None = None
    ) -> str:
        memory_block = ""
        avoid_block = ""
        if memory:
            memory_block = f"Previous pages visited: {', '.join(memory)}\n\n"
            avoid_block = "- Do not suggest any pages that have already been visited in this path.\n\n" if memory else ""
        return f"""
        You are playing the Wikispeedia game.
        Current page: "{current}"
        Goal page: "{goal}"
        {memory_block}
        Think step-by-step and suggest exactly 3 possible next Wikipedia pages that could help reach the goal.
        - Only choose from this list of valid outgoing links: {valid_links}
        - For each suggestion, include:
          * next_page (must be exactly one of the valid_links)
          * rating (1–10, higher = more promising)
        {avoid_block} 
        Here is some additional external knowledge that might help you make better decisions:
        {external_knowledge}, you can use this information to inform your suggestions.
        Return your answer strictly following this JSON schema:
        {self._get_response_json_schema()}
        """

