import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import plotly.graph_objects as go
import plotly.io as pio
import ast
from common import IOMixin
from config_local import Config

import seaborn as sns
sns.color_palette("tab10")  # Use same color palette as matplotlieb : categorical shifts plots


class Visualizer(IOMixin):
    def __init__(self, config: Config) -> None:
        super().__init__(config.results_folder)
        self.config = config

        # Hardcoded colors
        self.colors = {
            "Humans": "#1f77b4",
            "One-shot": "#ff7f0e",
            "CoT": "#2ca02c",
            "CoT(KB)": "#d62728",
            "ToT": "#9467bd"
        }

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

    def visualize_sankey_plot(self, cat_shift_df_pairs: pd.DataFrame, filename, start_cat, end_cat: str, label):
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
            title_text=f"Categorical Shifts from '{start_cat}' to '{end_cat}' Along Navigation Paths for {label}",
            height=700,
            width=1180,
        )

        # Plot
        fig.write_image(self.gen_output_path(filename))  # Save as static image
        #fig.show() # Uncomment and run for an interactive sankey diagram visualization

    def visualize_categorical_shifts_barchart_comparison(self, avg_shifts_path, avg_shifts_step, labels):
        # Convert any lists/arrays to scalar means
        avg_shifts_path_scalar = [np.mean(v) if isinstance(v, (list, np.ndarray)) else v for v in avg_shifts_path]
        avg_shifts_step_scalar = [np.mean(v) if isinstance(v, (list, np.ndarray)) else v for v in avg_shifts_step]
        
        # Now sort with labels
        avg_shifts_path_sorted, labels_path_sorted = map(list, zip(*sorted(zip(avg_shifts_path_scalar, labels))))
        avg_shifts_step_sorted, labels_step_sorted = map(list, zip(*sorted(zip(avg_shifts_step_scalar, labels))))
        
        # X-axis positions
        x_path = np.arange(len(labels_path_sorted))
        x_step = np.arange(len(labels_step_sorted))
        # Bar width
        width = 0.60
    
        # Map each label to its pre-defined color
        colors_path = [self.colors[label] for label in labels_path_sorted]
        colors_step = [self.colors[label] for label in labels_step_sorted]

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
        colors = [self.colors[label] for label in labels]

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
                color=self.colors[label]
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
        colors = [self.colors[label] for label in labels]
        
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
        colors = [self.colors[label] for label in labels]
        
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
    
    def visualize_categorical_shifts_mean_stepwise(self, mean_shifts_dict):
        plt.figure(figsize=(12, 6))
    
        # Counter to track all 1.0 lines
        all_ones_counter = 0
        jitter_step = 0.0019  
    
        for name, values in mean_shifts_dict.items():
            if len(values) == 0:
                continue
    
            progress_axis = np.linspace(0, 1, len(values))
            values_to_plot = np.array(values, dtype=float)

            # add jitter
            if np.all(values_to_plot == 1):
                offset = ((all_ones_counter + 1) // 2) * jitter_step
                offset *= 1 if all_ones_counter % 2 else -1  # alternate 
            
                values_to_plot = values_to_plot.copy() + offset
                plt.plot(progress_axis, values_to_plot, linewidth=1.8, label=name, color=self.colors[name])
            
                all_ones_counter += 1
            else:
                plt.plot(progress_axis, values_to_plot, linewidth=1.8, label=name, color=self.colors[name])
                
        plt.xlabel("Normalized Progress (0–1) for Start → Destination")
        plt.ylabel("Stepwise Mean Categorical Shifts")
        plt.title("Stepwise Mean Categorical Shifts Along Normalized Paths")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def visualize_stepwise_mean_similarity(self, mean_similarity, progress_axis):
        plt.figure(figsize=(12, 6))
    
        for name, values in mean_similarity.items():
            plt.plot(progress_axis, values, linewidth=2, label=name, color=self.colors[name])
    
        plt.xlabel("Normalized Progress (0–1) for Start → Destination")
        plt.ylabel("Mean Semantic Similarity")
        plt.title("Mean Semantic Similarity Along Normalized Paths")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def visualize_stepwise_mean_information_gain(self, information_gains, progress_axis, num_bins=10):
        players = information_gains.keys()

        x_bins = np.linspace(0, 1, num_bins)
        x = np.arange(num_bins)  # positions for grouped bars

        total_groups = len(players)
        bar_width = 0.6 / total_groups
        offsets = np.linspace(-((total_groups - 1) / 2) * bar_width, ((total_groups - 1) / 2) * bar_width, total_groups)

        plt.figure(figsize=(12, 6))
        bin_idx = np.digitize(progress_axis, x_bins) - 1

        for i, name in enumerate(players):
            vectors = information_gains[name]

            vals = np.zeros(num_bins)
            for b in range(num_bins):
                vals[b] = np.mean(vectors[bin_idx == b])

            plt.bar(x + offsets[i], vals, width=bar_width, label=name, color=self.colors[name], alpha=0.85)

        plt.xticks(x, [f'{p:.2f}' for p in x_bins], rotation=0)
        plt.xlabel('Normalized Progress (0–1) for Start → Destination')
        plt.ylabel('Stepwise Mean Information Gain')
        plt.title('Stepwise Mean Information Gain Along Normalized Paths')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4, axis='y')
        plt.tight_layout()
        plt.show()

    def plot_stepwise_similarity_violin(self, stepwise_similarity_flat):
        plt.figure(figsize=(10, 6))

        players = list(stepwise_similarity_flat.keys())
        data = [stepwise_similarity_flat[p] for p in players]

        parts = plt.violinplot(data, showmeans=True, showmedians=True)

        # Color each violin
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.colors.get(players[i], "#888888"))  # fallback
            pc.set_edgecolor('black')
            pc.set_alpha(0.6)

        plt.xticks(np.arange(1, len(players)+1), players)
        plt.ylabel("Stepwise Semantic Similarity (cosine similarity)")
        plt.title("Distribution of Stepwise Semantic Similarity per Player")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_path_length_distribution(self, groups):
        plt.figure(figsize=(10, 6))

        players = list(groups.keys())
        data = []

        for name in players:
            paths = groups[name]
            # Compute lengths
            lengths = [len(p) for p in paths]
            # Limit humans paths to 11
            if name == "Humans":
                lengths = [min(l, 11) for l in lengths]
            data.append(lengths)

        parts = plt.violinplot(data, showmeans=True, showmedians=True)

        # Color violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.colors.get(players[i], "#888888"))
            pc.set_edgecolor('black')
            pc.set_alpha(0.6)

        plt.xticks(np.arange(1, len(players)+1), players)
        plt.ylabel("Path Length (number of steps)")
        plt.title("Distribution of Path Lengths per Player")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_path_length_distribution_horizontal(self, groups):
        plt.figure(figsize=(10, 6))

        players = list(groups.keys())
        data = []

        for name in players:
            paths = groups[name]
            lengths = [len(p) for p in paths]
            # limit Humans to 11
            if name == "Humans":
                lengths = [min(l, 11) for l in lengths]  
            data.append(lengths)

        # Horizontal violin plot
        parts = plt.violinplot(data, showmeans=True, showmedians=True, vert=False)

        # Color violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.colors.get(players[i], "#888888"))
            pc.set_edgecolor('black')
            pc.set_alpha(0.6)

        plt.yticks(np.arange(1, len(players)+1), players)
        plt.xlabel("Path Length (No. of steps)")
        plt.title("Distribution of Path Lengths per Player")
        plt.grid(True, linestyle="--", alpha=0.3)
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

    def visualize_success_rate(self,llm_df: pd.DataFrame, human_df:pd.DataFrame):
        llm_path_stats = llm_df.copy()
        agg_df = human_df.groupby(["start", "destination"]).agg(
            success_count=("end", lambda x: x.notnull().sum()),
            sample_count=("end", "size"),
            path_length=("path", lambda x: x.dropna().map(len).mean())
        ).reset_index()

        # success rate
        agg_df["success_rate"] = agg_df["success_count"] / agg_df["sample_count"]
        # select all the  pairs in my llm_path_stats
        human_success_rates = []
        for idx, row in llm_path_stats.iterrows():
            start = row['start']
            destination = row['destination']
            match = agg_df[(agg_df['start'] == start) & (agg_df['destination'] == destination)]
            if not match.empty:
                human_success_rates.append(match['success_rate'].values[0])
            else:
                human_success_rates.append(None)
        llm_path_stats['human_success_rate'] = human_success_rates
        llm_summary = llm_path_stats.groupby(['model', 'difficulty']).agg(
            link_aware_success_rate=('link_aware_success', 'mean'),
            external_success_rate=('external_info_success', 'mean'),
            tot_success_rate=('tot_success', 'mean'),
            count=('link_aware_success', 'size'),
            human_success_rate_avg=('human_success_rate', 'mean')
        ).reset_index()

        models_order = ['oss_20b', 'oss_120b']

        all_difficulties = sorted(llm_summary['difficulty'].dropna().unique().tolist(), key=lambda x: (str(x)))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # human success rate not a bar, but a horizontal line for each difficulty level for each model
        for i, model in enumerate(models_order):
            ax = axes[i]
            group = llm_summary[llm_summary['model'] == model]
            if group.empty:
                ax.axis('off')
                continue
            group.plot(
                x='difficulty',
                y=['link_aware_success_rate', 'external_success_rate', 'tot_success_rate'],
                kind='bar',
                ax=ax,
                title=f'Success Rate by Difficulty - {model}',
                ylabel='Success Rate',
                xlabel='Difficulty Level',
                ylim=(0, 1),
                legend=True
            )
            for _, row in group.iterrows():
                difficulty = row['difficulty']
                human_rate = row['human_success_rate_avg']
                if pd.notna(human_rate):
                    ax.hlines(y=human_rate, xmin=all_difficulties.index(difficulty) - 0.4,
                              xmax=all_difficulties.index(difficulty) + 0.4, colors='red', linestyles='dashed',
                              label='Human Success Rate' if difficulty == all_difficulties[0] else "")

            handles, labels = ax.get_legend_handles_labels()
            new_labels = ['Human', 'CoT', 'CoT(KB)', 'ToT']
            ax.legend(handles, new_labels)
        plt.tight_layout()
        plt.show()

    def visualize_time_usage(self, llm_df: pd.DataFrame):
        llm_path_stats = llm_df.copy()
        llm_path_stats['blind_time'] = llm_path_stats['blind_input_tokens'] + llm_path_stats['blind_output_tokens']
        llm_path_stats['link_aware_time'] = llm_path_stats['link_aware_input_tokens'] + llm_path_stats[
            'link_aware_output_tokens']
        llm_path_stats['external_time'] = llm_path_stats['external_info_input_tokens'] + llm_path_stats[
            'external_info_output_tokens']
        llm_path_stats['tot_time'] = llm_path_stats['tot_input_tokens'] + llm_path_stats['tot_output_tokens']
        llm_time_summary = llm_path_stats.groupby(['model', 'difficulty']).agg(
            blind_time_avg=('blind_time', 'mean'),
            link_aware_time_avg=('link_aware_time', 'mean'),
            external_time_avg=('external_time', 'mean'),
            tot_time_avg=('tot_time', 'mean'),
        ).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        models_order = ['oss_20b', 'oss_120b']
        for i, model in enumerate(models_order):
            ax = axes[i]
            group = llm_time_summary[llm_time_summary['model'] == model]
            if group.empty:
                ax.axis('off')
                continue
            group.plot(
                x='difficulty',
                y=['blind_time_avg', 'link_aware_time_avg', 'external_time_avg', 'tot_time_avg'],
                kind='bar',
                ax=ax,
                title=f'Time Usage by Difficulty - {model}',
                ylabel='Average Token Usage',
                xlabel='Difficulty Level',
                legend=True
            )
            handles, labels = ax.get_legend_handles_labels()
            new_labels = ['Blind','CoT', 'CoT(KB)', 'ToT']
            ax.legend(handles, new_labels)
        plt.tight_layout()
        plt.show()

    def visualize_path_length(self, llm_df: pd.DataFrame, human_df: pd.DataFrame):
        def is_valid_path(path, dest):
            return path is not None and len(path) > 0 and path[-1] == dest
        def compute_steps(path, dest):
            if is_valid_path(path, dest):
                return len(path)
            else:
                return np.nan
        llm_path_stats = llm_df.copy()
        llm_path_stats['link_aware_steps'] = llm_path_stats.apply(
            lambda row: compute_steps(row['link_aware_paths'], row['destination']), axis=1)
        llm_path_stats['external_info_steps'] = llm_path_stats.apply(
            lambda row: compute_steps(row['external_info_paths'], row['destination']), axis=1)
        llm_path_stats['tot_steps'] = llm_path_stats.apply(lambda row: compute_steps
        (row['tot_paths'], row['destination']), axis=1)
        avg_steps_list = []
        agg_df = human_df.groupby(["start", "destination"]).agg(
            success_count=("end", lambda x: x.notnull().sum()),
            sample_count=("end", "size"),
            path_length=("path", lambda x: x.dropna().map(len).mean())
        ).reset_index()
        for idx, row in llm_path_stats.iterrows():
            start = row['start']
            destination = row['destination']
            match = agg_df[(agg_df['start'] == start) & (agg_df['destination'] == destination)]
            if not match.empty:
                avg_steps_list.append(match['path_length'].values[0])
            else:
                avg_steps_list.append(None)
        llm_path_stats['human_avg_steps'] = avg_steps_list
        llm_length_summary = llm_path_stats.groupby(['model', 'difficulty']).agg(
            link_aware_length_avg=('link_aware_steps', 'mean'),
            external_length_avg=('external_info_steps', 'mean'),
            tot_length_avg=('tot_steps', 'mean'),
            human_avg_steps_avg=('human_avg_steps',
                                 'mean')
        ).reset_index()
        models_order = ['oss_20b', 'oss_120b']
        all_difficulties = sorted(llm_length_summary['difficulty'].dropna().unique().tolist(), key=lambda x: (str(x)))
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        for i, model in enumerate(models_order):
            ax = axes[i]
            group = llm_length_summary[llm_length_summary['model'] == model]
            if group.empty:
                ax.axis('off')
                continue
            group.plot(
                x='difficulty',
                y=['link_aware_length_avg', 'external_length_avg', 'tot_length_avg'],
                kind='bar',
                ax=ax,
                title=f'Path Length by Difficulty - {model}',
                ylabel='Average Path Length',
                xlabel='Difficulty Level',
                legend=True
            )
            for _, row in group.iterrows():
                difficulty = row['difficulty']
                human_length = row['human_avg_steps_avg']
                if pd.notna(human_length):
                    ax.hlines(y=human_length, xmin=all_difficulties.index(difficulty) - 0.4,
                              xmax=all_difficulties.index(difficulty) + 0.4, colors='red', linestyles='dashed',
                              label='Human Average Steps' if difficulty == all_difficulties[0] else "")

            handles, labels = ax.get_legend_handles_labels()
            new_labels = ['Human', 'CoT', 'CoT(KB)', 'ToT']
            ax.legend(handles, new_labels)
        plt.tight_layout()
        plt.show()


    def visualize_specific_semantic_space(self,metadata: pd.DataFrame, emb_path: str, article_titles: list[str]):
        embeddings = torch.load(emb_path, map_location="cpu")
        articles_df = metadata
        embeddings_np = embeddings.detach().cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        embeddings_2d = tsne.fit_transform(embeddings_np)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, color='tab:blue')
        max_labels = 100
        indices_to_label = [i for i, title in enumerate(metadata['article']) if title in article_titles]
        for i in indices_to_label:
            ax.text(
                embeddings_2d[i, 0] + 0.5,
                embeddings_2d[i, 1] + 0.5,
                s=metadata['article'].iloc[i],
                ha='left', va='bottom', fontsize=8
            )
            ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=100, color='tab:red')
            # and link them with lines
            # the line didnt match the position, because
        for j in range(len(indices_to_label) - 1):
            i1 = indices_to_label[j]
            i2 = indices_to_label[j + 1]
            # linestyle is not dashed, but solid
            ax.plot([embeddings_2d[i1, 0], embeddings_2d[i2, 0]], [embeddings_2d[i1, 1], embeddings_2d[i2, 1]],
                    color='tab:green', linewidth=5, linestyle='solid', alpha=0.5)

        # title show the path
        ax.set_title('Path: ' + ' -> '.join(article_titles))
        plt.tight_layout()
        plt.show()
