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

    def get_config(self, debug: bool) -> dict:
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
                    'schema': self._get_response_json_schema()
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
                        'parameters': self._get_response_json_schema()
                    }
                }],
                'tool_choice': {
                    'type': 'function',
                    'function': {'name': 'navigation_step'}
                }
            }
            base_config.update(json_force_config)
        return base_config

    def generate_prompt(self, current: str, goal: str, valid_links: list[str]) -> str:
        return f"""
        You are playing the Wikispeedia game.
        Current page: "{current}"
        Goal page: "{goal}"
    
        Think step-by-step and suggest exactly 3 possible next Wikipedia pages that could help reach the goal.
        - Only choose from this list of valid outgoing links: {valid_links}
        - For each suggestion, include:
          * next_page (must be exactly one of the valid_links)
          * reason (why this might be useful)
          * rating (1–10, higher = more promising)
    
        Return your answer strictly following the JSON schema provided.
        """


