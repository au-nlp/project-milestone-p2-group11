from dataclasses import dataclass, field


@dataclass
class InputFile:
    source: str
    path: str


@dataclass
class Config:
    header_pattern: str
    results_folder: str
    embeddings_folder: str
    sentence_transformers_checkpoint: str
    input_file_articles: str 
    information_gain_num_bins: int
    input_files_paths: list[InputFile] = field(default_factory=list)


config = Config(
    header_pattern='# FORMAT:',
    results_folder='results/',
    embeddings_folder='article_emb/',
    sentence_transformers_checkpoint='sentence-transformers/all-MiniLM-L6-v2',
    input_file_articles='data/wikispeedia_paths-and-graph/articles.tsv',
    information_gain_num_bins=10,
    input_files_paths=[
        InputFile(source='finished', path='data/wikispeedia_paths-and-graph/paths_finished.tsv'),
        InputFile(source='unfinished', path='data/wikispeedia_paths-and-graph/paths_unfinished.tsv'),
    ],
)
