import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import plotly.graph_objects as go
import plotly.io as pio

from common import IOMixin
from config_local import Config

import seaborn as sns
sns.color_palette("tab10")  # Use same color palette as matplotlieb : categorical shifts plots

# Define a fixed color mapping for the categorical shift plots
cat_color_map = {
    "Humans": "#1f77b4",
    "One-shot": "#ff7f0e",
    "CoT": "#2ca02c",
    "CoT(KB)": "#d62728",
    "ToT": "#9467bd"
}

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
        #embeddings = torch.load(emb_path, weights_only=True)
        embeddings = torch.load(emb_path, map_location="cpu")
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

    def visualize_category_distribution_for_paths(self, df: pd.DataFrame, filename: str):
        # Plot the 15 top categories while excluding missing categories
        fig, _ = plt.subplots(figsize=(12, 8))
        plt.barh(df.head(15)["category"], df.head(15)["count"])
        plt.xlabel("Count")
        plt.title("Global Category Distribution for Navigation Paths (Top 15)")
        plt.gca().invert_yaxis()
        plt.show()
        fig.savefig(self.gen_output_path(filename), dpi=300)

    def visualize_top_categorical_shifts(self, df: pd.DataFrame, filename: str):
        # Plot the most frequent categorical shifts
        fig, _ = plt.subplots(figsize=(12, 8))
        plt.barh(df["from_cat"] + " -> " + df["to_cat"], df["count"])
        plt.xlabel("Count")
        plt.title("Most Frequent Categories Pairs Along Navigation Paths (Top 15)")
        plt.gca().invert_yaxis()
        plt.show()
        fig.savefig(self.gen_output_path(filename), dpi=300)

    def visualize_top_pairwise_categorical_shifts(self, df: pd.DataFrame, filename: str):
        # Plot the most frequent categorical shifts
        fig, _ = plt.subplots(figsize=(12, 8))
        plt.barh(df["from_cat"] + " -> " + df["to_cat"], df["count"])
        plt.xlabel("Count")
        plt.title("Most Frequent Pairwise Category Shifts Along Navigation Paths (Top 15)")
        plt.gca().invert_yaxis()
        plt.show()
        fig.savefig(self.gen_output_path(filename), dpi=300)

    def visualize_sankey_plot(self, cat_shift_df_pairs: pd.DataFrame, filename, start_cat, end_cat: str):
        # Sankey diagram
        # Reference: https://plotly.com/python/sankey-diagram/
        # Render method
        pio.renderers.default = "notebook_connected"

        # Extract unique category nodes 
        nodes = pd.unique(cat_shift_df_pairs[["cat_from", "cat_to"]].values.ravel())
        # remove positional suffix (e.g. country_1)
        node_labels = [n.rsplit("_", 1)[0] for n in nodes] # type: ignore
        # Define node indices
        node_indices = {n: i for i, n in enumerate(nodes)}

        # Build diagram
        fig = go.Figure(go.Sankey(
            # Nodes
            node=dict(label=node_labels, 
                      pad=22, 
                      thickness=16
                     ),
            # Links
            link=dict(
                source=[node_indices[a] for a in cat_shift_df_pairs["cat_from"]],
                target=[node_indices[b] for b in cat_shift_df_pairs["cat_to"]],
                value=cat_shift_df_pairs["count"].tolist(),
            )
        ))

        # Layout
        fig.update_layout(
            title_text=f"Categorical drift from '{start_cat}' -> '{end_cat}'",
            height=700,
            width=1180,
        )

        # Plot
        fig.write_image(self.gen_output_path(filename))  # Save as static image
        #fig.show() # Uncomment and run for an interactive sankey diagram visualization

    def visualize_categorical_shifts_barchart_comparison(self, avg_shifts_path, avg_shifts_step, labels):
        # Sort navigation paths shifts
        avg_shifts_path_sorted, labels_path_sorted = zip(*sorted(zip(avg_shifts_path, labels)))
        x_path = np.arange(len(labels_path_sorted)) # convert labels_path_sorted to Numpy array
        
        # Sort navigation steps shifts
        avg_shifts_step_sorted, labels_step_sorted = zip(*sorted(zip(avg_shifts_step, labels)))
        x_step = np.arange(len(labels_step_sorted)) # convert labels_step_sorted to Numpy array

        # Bar width
        width = 0.60
    
        # Map each label to its pre-defined color
        colors_path = [cat_color_map[label] for label in labels_path_sorted]
        colors_step = [cat_color_map[label] for label in labels_step_sorted]

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
        # Left plot: navigation paths shifts
        ax1.bar(x_path, avg_shifts_path_sorted, width, color=colors_path)
        ax1.set_ylabel('Average categorical shift')
        ax1.set_title('Average Categorical Shifts per Navigation Path')
        ax1.set_xticks(x_path)
        ax1.set_xticklabels(labels_path_sorted, rotation=45, ha='right')
    
        # Right plot: navigation steps shifts
        rects2 = ax2.bar(x_step, avg_shifts_step_sorted, width, color=colors_step)
        ax2.set_ylabel('Proportion of consecutive article pairs with shift')
        ax2.set_title('Average Categorical Shifts per Step')
        ax2.set_xticks(x_step)
        ax2.set_xticklabels(labels_step_sorted, rotation=45, ha='right')
        ax2.legend(rects2, labels_step_sorted, title="Player", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
 
    def visualize_categorical_shifts_violin_comparison(self, shifts_per_path_list, labels):
        # Map each label to its pre-defined color
        colors = [cat_color_map[label] for label in labels]

        # Plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=shifts_per_path_list, palette=colors)
        plt.xticks(range(len(labels)), labels)
        plt.ylabel('No. of categorical shifts')
        plt.title('Violin Chart of Categorical Shifts Distribution (Navigation Path)')
        plt.show()

    def visualize_categorical_shifts_focuses_violin_comparison(self, shifts_per_path_list, labels):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
        # Plot all violins at x-pos
        for i, (shifts, label) in enumerate(zip(shifts_per_path_list, labels)):
            sns.violinplot(
                x=[i] * len(shifts),
                y=shifts,
                ax=ax,
                cut=0,
                density_norm='width',
                color=cat_color_map[label]
            )
    
        ax.set_title('Violin Chart of Categorical Shift Distributions (Navigation Path)')
        ax.set_ylabel('No. of categorical shifts')
        ax.set_ylim(0, 35)
        ax.grid(axis='y', linestyle='-', color='gray', alpha=0.2)
        ax.set_yticks(list(np.arange(0, 16, 1)) + [15, 20, 25, 30, 35])
        ax.set_xticks(list(range(len(labels))))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    def visualize_categorical_shifts_scatterplot_comparison(self, shifts_per_path_list, labels):
        # Map each label to its pre-defined color
        colors = [cat_color_map[label] for label in labels]
        
        # Combine all shifts into a single DataFrame for seaborn
        df_list = []
        for shifts, label in zip(shifts_per_path_list, labels):
            df_list.append(pd.DataFrame({
                'Shifts_per_path': shifts,
                'Group': [label]*len(shifts)
            }))
        df = pd.concat(df_list, ignore_index=True)
    
        plt.figure(figsize=(12, 6))
        sns.stripplot(
            x='Group',
            y='Shifts_per_path',
            hue='Group',
            data=df,
            jitter=0.35,
            size=7,
            alpha=0.4,
            palette=colors,
            dodge=False,
            legend=False  # disable legend
        )
        plt.title('Scatterplot of Categorical Shift Distributions (Navigation Path)')
        plt.xlabel('')
        plt.ylabel('No. of categorical shifts')
        plt.ylim(0)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    def visualize_categorical_shifts_scatterplot_sidebyside_comparison(self, shifts_per_path_list, labels):
        # Map each label to its pre-defined color
        colors = [cat_color_map[label] for label in labels]
        
        # Combine all shifts into a single DataFrame for seaborn
        df_list = []
        for shifts, label in zip(shifts_per_path_list, labels):
            df_list.append(pd.DataFrame({
                'Shifts_per_path': shifts,
                'Group': [label]*len(shifts)
            }))
        df = pd.concat(df_list, ignore_index=True)

        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(18, 8), sharex=True)
        # Y axis scope for each sublot
        y_limits = [None, 50, 20, 8]
        titles = ['All datapoints', 'Path Lengths: 0-100', 'Path Lengths: 0-50', 'Path Lengths: 0-20']
        for i, (ax, ylim, title) in enumerate(zip(axes, y_limits, titles)):
            sns.stripplot(
                x='Group',
                y='Shifts_per_path',
                data=df,
                hue='Group',
                palette=colors,
                dodge=False,
                jitter=0.35,
                size=7,
                alpha=0.4,
                ax=ax,
                legend=False
            )
            ax.set_title(f'Categorical Shifts Distribution ({title})')
            ax.set_xlabel('')       # remove x-axis label
            ax.set_ylabel('No. of categorical shifts') 
            # Hide y-axis ticks and labels for all except the first subplot
            if i != 0:
                ax.set_ylabel('')
            ax.set_ylim(0, ylim)
            ax.grid(axis='y', linestyle='-', color='gray', alpha=0.2)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def visualize_ratings_distribution(self, df: pd.DataFrame, filename: str):
        fig, _ = plt.subplots(figsize=(12, 8))
        df['rating'].hist(bins=50, alpha=0.5, label='finished', color='blue', range=(0,5))
        plt.title('Distribution of the ratings of the finished paths')
        plt.xlabel('Rating')
        plt.ylabel('Number of paths')
        plt.legend()
        plt.show()
        fig.savefig(self.gen_output_path(filename), dpi=300)

    def visualize_duration_distribution_unfinished(self, df: pd.DataFrame, filename: str):
        fig, _ = plt.subplots(figsize=(12, 8))
        df[df['source'] == 'unfinished']['durationInSec'].hist(bins=50) # type: ignore
        plt.title('Distribution of the durations of the unfinished paths')
        plt.xlabel('Duration in seconds')
        plt.ylabel('Number of paths')
        plt.show()
        fig.savefig(self.gen_output_path(filename), dpi=300)

    def visualize_duration_distribution_unfinished_timeout(self, df: pd.DataFrame, filename: str):
        fig, _ = plt.subplots(figsize=(12, 8))
        df[(df['source'] == 'unfinished') & (df['type'] == 'timeout')]['durationInSec'].hist(bins=50) # type: ignore
        plt.title('Distribution of the duration of the timeout paths')
        plt.xlabel('Duration in seconds')
        plt.ylabel('Number of paths')
        plt.show()
        fig.savefig(self.gen_output_path(filename), dpi=300)
