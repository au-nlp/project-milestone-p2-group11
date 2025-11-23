import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Literal


load_dotenv()


@dataclass
class InputFile:
    source: str
    path: str


@dataclass
class LLMConfig:
    model: str
    max_completion_tokens: int
    reasoning_effort: Literal['high', 'medium', 'low']
    top_p: int
    temperature: int
    stream: bool
    stop: str | None


@dataclass
class Config:
    debug: bool
    header_pattern: str
    results_folder: str
    embeddings_folder: str
    sentence_transformers_checkpoint: str
    input_file_articles: str
    input_file_categories: str
    input_file_links: str
    max_number_of_guesses_by_llm: int
    information_gain_num_bins: int
    base_branch: int
    max_depth: int
    groq_api_key: str
    digital_ocean_api_key: str
    digital_ocean_url: str
    llm_config: LLMConfig
    input_files_paths: list[InputFile] = field(default_factory=list)

# Digitalocean models:
#     - openai-gpt-oss-120b
#     - openai-gpt-oss-20b
#     - llama3-8b-instruct
#     - llama3.3-70b-instruct

config = Config(
    debug=False,

    # step 0
    header_pattern='# FORMAT:',
    results_folder='results/',
    embeddings_folder='article_emb/',
    sentence_transformers_checkpoint='sentence-transformers/all-MiniLM-L6-v2',
    input_file_articles='data/wikispeedia_paths-and-graph/articles.tsv',
    input_file_categories='data/wikispeedia_paths-and-graph/categories.tsv',
    input_file_links='data/wikispeedia_paths-and-graph/links.tsv',
    information_gain_num_bins=10,
    input_files_paths=[
        InputFile(source='finished', path='data/wikispeedia_paths-and-graph/paths_finished.tsv'),
        InputFile(source='unfinished', path='data/wikispeedia_paths-and-graph/paths_unfinished.tsv'),
    ],

    # step 1
    groq_api_key=os.getenv('GROQ_API_KEY', ''),
    digital_ocean_api_key=os.getenv('DIGITAL_OCEAN_API_KEY', ''),
    digital_ocean_url='https://inference.do-ai.run/v1/',
    max_number_of_guesses_by_llm=10,
    base_branch=3,
    max_depth=14,
    llm_config=LLMConfig(
        # model='openai/gpt-oss-20b',  # Groq
        model='openai-gpt-oss-20b',
        max_completion_tokens=2048,
        reasoning_effort='medium',
        top_p=1,
        temperature=1,
        stream=False,
        stop=None,
    ),
)
