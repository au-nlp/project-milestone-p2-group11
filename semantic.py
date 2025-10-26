from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

from config_local import Config
from common import IOMixin


class SemanticAnalyser(IOMixin):
    def __init__(self, config: Config) -> None:
        super().__init__(config.results_folder, config.embeddings_folder)
        self.model = SentenceTransformer(config.sentence_transformers_checkpoint)
        self.config = config

        self.metadata_filename = 'metadata.csv'
        self.embeddings_filename = 'embeddings.pt'

    def _get_metadata_mapping(self) -> dict[str, int]:
        metadata = pd.read_csv(self.gen_output_path(self.metadata_filename))
        return {v: k for k, v in metadata.to_dict()['article'].items()}

    def _get_embeddings(self) -> torch.Tensor:
        return torch.load(self.gen_output_path(self.embeddings_filename))

    def generate_embeddings(self, df: pd.DataFrame):
        # save metadata.csv with the article names -> so we can lookup embeddings.npy
        df.to_csv(self.gen_output_path(self.metadata_filename), index=False)

        # compute embedding for both lists
        embeddings = self.model.encode(df['article'], convert_to_tensor=True) # type: ignore
        torch.save(embeddings, self.gen_output_path(self.embeddings_filename))

    def get_avg_information_gain(self, paths: pd.Series) -> list[np.floating]:
        # get metadata mapping (title -> index) in cache
        metadata_cache = self._get_metadata_mapping()
        embeddings = self._get_embeddings()
        num_bins = self.config.information_gain_num_bins

        # track information gains by bins -> controlled by num_bins
        information_gains = defaultdict(list)
        for path in paths:
            if len(path) < 2:
                continue

            # get start and destination embeddings
            start_index = metadata_cache[path[0]]
            destination_index = metadata_cache[path[-1]]
            start_embedding = embeddings[start_index]
            destination_embedding = embeddings[destination_index]
            
            # iterate over finished paths one by one
            for i in range(1, len(path)):
                # get current page embedding
                curr_page = path[i]
                curr_page_index = metadata_cache[curr_page]
                curr_embedding = embeddings[curr_page_index]

                # calculate bin_index
                relative_position = i / (len(path) - 1)
                bin_index = int(relative_position * (num_bins - 1)) 

                # compute cosine similarity to start and to destination
                sim_to_start = util.pytorch_cos_sim(curr_embedding, start_embedding).item()
                sim_to_destination = util.pytorch_cos_sim(curr_embedding, destination_embedding).item()

                # calculate information gain
                information_gain = sim_to_destination - sim_to_start
                information_gains[bin_index].append(information_gain)

        # average across all paths for each bin
        return [np.mean(information_gains[i]) for i in sorted(information_gains.keys())]
