import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

from common import IOMixin
from config_local import Config


class Visualizer(IOMixin):
    def __init__(self, config: Config) -> None:
        super().__init__(config.results_folder)
        self.config = config

    def visualize_top_k(self, df: pd.DataFrame, filename: str):
        df['path_label'] = df['start'] + ' -> ' + df['destination']

        pivot_df = df.pivot_table(
            index='path_label',
            columns='source',
            values='sample_count',
            fill_value=0
        )

        pivot_df['total'] = pivot_df['finished'] + pivot_df['unfinished']
        pivot_df = pivot_df.sort_values('total', ascending=False)

        labels = pivot_df.index
        finished_counts = pivot_df['finished'] / pivot_df['total'] * 100
        unfinished_counts = pivot_df['unfinished'] / pivot_df['total'] * 100

        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.bar(labels, finished_counts, label='finished')
        bars2 = ax.bar(labels, unfinished_counts, bottom=finished_counts, label='unfinished')

        for i, (bar, count) in enumerate(zip(bars1, pivot_df['finished'])):
            height = bar.get_height()
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height/2,
                    f'{int(count)}',
                    ha='center', 
                    va='center', 
                )
                
        for i, (bar, count) in enumerate(zip(bars2, pivot_df['unfinished'])):
            height = bar.get_height()
            bottom = finished_counts.iloc[i]
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    bottom + height/2,
                    f'{int(count)}',
                    ha='center', 
                    va='center', 
                )

        ax.set_title('Sample distribution by start-destination path')
        ax.set_ylabel('Percentage')
        ax.legend(loc='upper right')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        fig.savefig(self.gen_output_path(filename), dpi=300)

    def visualize_top_k_avg_hops(self, df: pd.DataFrame, filename: str):
        filtered_top_k_df = df[df['source'] == 'finished']
        fig, ax = plt.subplots(figsize=(12, 8))
        for row in filtered_top_k_df.iterrows():
            x = row[1]['avg_num_hops']
            y = row[1]['avg_rating']
            size = row[1]['sample_count']
            label: str = row[1]['path_label'] # type: ignore

            ax.scatter(x, y, s=np.log(size) * 10, color='tab:blue')
            ax.text(x + .1, y - .2, s=label, ha='left', va='bottom', rotation=-10, fontsize=8)

        ax.set_title('Average number of hops vs average rating')
        ax.set_ylabel('Average rating')
        ax.set_xlabel('Average number of hops')
        plt.tight_layout()
        plt.show()
        fig.savefig(self.gen_output_path(filename), dpi=300)

    def visualize_hop_distribution(
            self, df: pd.DataFrame, filename: str, xlim: tuple, start: str, destination: str
    ):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(df.index, df.values)
        ax.set_title(f'Hop distribution for {start} -> {destination}\nxlim={xlim}')
        ax.set_ylabel('Number of runs')
        ax.set_xlabel('Number of hops')
        ax.set_xlim(xlim)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        fig.savefig(self.gen_output_path(filename), dpi=300)

    def visualize_information_gain(self, data: list[np.floating], filename: str):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(np.linspace(0, 1, len(data)), data, width=1/self.config.information_gain_num_bins)
        ax.set_title('Average information gain vs progress of navigation')
        ax.set_ylabel('Average information gain')
        ax.set_xlabel('Progress of the navigation')
        plt.tight_layout()
        plt.show()
        fig.savefig(self.gen_output_path(filename), dpi=300)

    def visualize_semantic_space(self, filename: str, metadata:pd.DataFrame, emb_path: str):
        embeddings = torch.load(emb_path, weights_only=True)
        articles_df = metadata
        embeddings_np = embeddings.detach().cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        embeddings_2d = tsne.fit_transform(embeddings_np)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, color='tab:blue')
        max_labels = 100
        indices_to_label = np.random.choice(len(metadata), size=min(max_labels, len(metadata)), replace=False)

        for i in indices_to_label:
            ax.text(
                embeddings_2d[i, 0] + 0.5,
                embeddings_2d[i, 1] + 0.5,
                s=metadata['article'].iloc[i],
                ha='left', va='bottom', fontsize=8
            )

        ax.set_title('t-SNE visualization of article embeddings')
        plt.tight_layout()
        plt.show()
        fig.savefig(self.gen_output_path(filename), dpi=300)
