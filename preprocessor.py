import pandas as pd
import numpy as np
import csv
import logging
from collections import defaultdict, Counter
import ast
import random

from config_local import Config
import urllib.parse
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, config: Config) -> None:
        self.config = config

    def _merge_dfs(self, *args: tuple[str, pd.DataFrame]) -> pd.DataFrame:
        for source, _df in args:
            _df['source'] = source  # annotate df according to source
        return pd.concat([df for _, df in args], ignore_index=True)

    def _annotate_start_and_end(self, df: pd.DataFrame):
        split_cols = df['path'].str.split(';')
        df['start'] = split_cols.str[0]
        df['end'] = split_cols.str[-1].where(df['source'] == 'finished')

    def _annotate_num_hops(self, df: pd.DataFrame):
        df['num_hops'] = df['path'].str.split(';').str.len()

    def _annotate_destination(self, df: pd.DataFrame):
        df['destination'] = np.where(
            df['source'] == 'finished',
            df['end'],
            df['target']
        )

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        self._annotate_start_and_end(df)
        self._annotate_num_hops(df)
        self._annotate_destination(df)
        return df

    def _filter_df(self, df: pd.DataFrame, start, destination, source: str) -> pd.DataFrame:
        filtered_df = df[
            (df['start'] == start) &
            (df['destination'] == destination) &
            (df['source'] == source)
        ]
        return pd.DataFrame(filtered_df)

    def _resolve_back_clicks(self, path: list):
        return [
           article for i, article in enumerate(path) 
           if article != '<' and (i == len(path) - 1 or path[i+1] != '<')
        ]

    def parse_input_file(self, path: str) -> pd.DataFrame | None:
        column_names, header_pattern = None, self.config.header_pattern

        with open(path, 'r', encoding='utf-8') as tsvfile:
            tsv_reader = csv.reader(tsvfile, delimiter='\t')
            for row in tsv_reader:  # streaming until FORMAT is extracted
                if len(row) > 0  and header_pattern in row[0]:
                    column_names = [el for el in row[0].rsplit(header_pattern)[-1].strip().split(' ') if el]
                    logger.info("Successfully retrieved the following headers: {column_names}")
                    break
                elif len(row) > 1:  # stop after comments
                    break

        if column_names is None:
            logger.error(f"Couldn't retrieve column_names in {path}")
            return
        
        df = pd.read_csv(path, comment='#', sep='\t', names=column_names)
        try:
            return df.map(urllib.parse.unquote)
        except TypeError:
            return df

    def get_merged_input_files(self) -> pd.DataFrame:
        # parsing input files into pd.DataFrames
        parsed_dfs = []
        for input_file in self.config.input_files_paths:
            df = self.parse_input_file(input_file.path)

            if df is not None:
                df['path'] = df['path'].apply(urllib.parse.unquote)

            if df is not None:
                parsed_dfs.append((input_file.source, df))

        # merge multiple dfs into a single one
        merged_df = self._merge_dfs(*parsed_dfs)

        # preprocess merged_df
        preprocessed_merged_df = self._preprocess_df(merged_df)
        return preprocessed_merged_df
    
    def get_top_paths_for_starts(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        # get top k start paths
        agg_with_start: pd.DataFrame = df.groupby(
            ['start', 'source'], 
            as_index=False,
        ).agg(
            sample_count=('hashedIpAddress', 'count'),
        ) # type: ignore

        target_groups_df = agg_with_start.nlargest(
            n=2*k,
            columns='sample_count',
        ).sort_values(
            by=['start', 'source']
        )
        
        # get detailed counts for both start - destination
        agg_with_start_and_end = df.groupby(
            ['start', 'source', 'destination'],
            as_index=False,
        ).agg(
            avg_rating=('rating', 'mean'),
            avg_num_hops=('num_hops', 'mean'),
            sample_count=('hashedIpAddress', 'count'),
        )
        
        # divide agg_weithith_start_and_end into finished and unfinished
        finished_paths: pd.DataFrame = (
            agg_with_start_and_end[agg_with_start_and_end['source'] == 'finished'] # type: ignore
        )
        unfinished_paths: pd.DataFrame = (
            agg_with_start_and_end[agg_with_start_and_end['source'] == 'unfinished'] # type: ignore
        )
        
        # merge start - destination on top k start paths
        top_finished_paths = pd.merge(
            finished_paths,
            target_groups_df[['start', 'source']],
            on=['start', 'source'],
            how='inner'
        )
        
        # filter out top 3 destinations for each of the start paths
        top_3_finished_dest = (
                top_finished_paths.sort_values('sample_count', ascending=False).groupby('start').head(3)
        )
        
        # selecting relevant values, later used in merge
        master_dest_list = top_3_finished_dest[['start', 'destination']]
        
        # merge top 3 destinations from finished paths to unfinished paths
        matching_unfinished_paths = pd.merge(
            unfinished_paths,
            master_dest_list,
            on=['start', 'destination'],
            how='inner' # This is the key: only keeps matching rows
        )
        
        # concat the finished and unfinished entries
        return pd.concat(
            [top_3_finished_dest, matching_unfinished_paths]).sort_values(['start', 'source', 'destination']
        )

    def get_hop_distribution(self, df: pd.DataFrame, start, destination: str) -> pd.Series:
        hop_distribution_df = self._filter_df(df, start, destination, 'finished').groupby(
            'num_hops'
        )['hashedIpAddress'].count()
        return pd.Series(hop_distribution_df)
    
    def get_parsed_paths(self, df: pd.DataFrame, start, destination: str) -> pd.Series:
        paths = self._filter_df(df, start, destination, 'finished')['path'].str.split(';')
        return pd.Series(paths.apply(self._resolve_back_clicks))

    def get_link_mappings(self, df: pd.DataFrame) -> defaultdict[str, list]:
        link_mappings = defaultdict(list)
        for _, row in df.iterrows():
            link_mappings[row['linkSource']].append(row['linkTarget'])
        return link_mappings

    def resolve_all_parsed_paths(self, df: pd.DataFrame) -> pd.DataFrame | None:
        if(type(df['path'].iloc[0]) == list):
            return
        df['path'] = df['path'].str.split(';').apply(self._resolve_back_clicks)
        df['num_hops'] = df['path'].str.len()
        return df
    
    def get_clean_categories_df(self, df: pd.DataFrame, categories: pd.DataFrame) -> tuple:
        # Only finished paths (perhaps a better idea for the salluviate plot?)
        df_finished = df[df["source"] == "finished"].copy()

        # Extract the category (final element) and create the article -> category mappings
        article_to_cat = (
            categories.assign(cat_short=categories["category"].str.split(".").str[-1])
            .groupby("article")["cat_short"]
            .apply(list)
            .to_dict()
        )

        # Map all articles in the navigation paths to its first category 
        # (mapping missing articles to 'unknown' as there is no category list available)
        article_cats = [
            article_to_cat.get(p, ["Unknown"])[0] for path in df["path"] for p in path.split(";")
        ]
        # For finished path (not flattened): used for the  alluviate plot
        article_cats_finished = [
            [article_to_cat.get(p, ["Unknown"])[0] for p in path.split(";")]for path in df_finished["path"]
        ]

        # Create dataframe using the mapped article caterogies
        cat_df = pd.DataFrame(
            {"category": article_cats}
        ).value_counts().reset_index(name="count").sort_values("count", ascending=False)

        # Remove missing articles
        cat_df_clean = cat_df[cat_df["category"] != "Unknown"].reset_index(drop=True)
        return cat_df_clean, cat_df, article_cats_finished, article_cats

    def get_category_shift_df(self, article_cats: pd.DataFrame, cat_df_clean: pd.DataFrame) -> pd.DataFrame:
        # Extract all consecitive category pairs (A → B) from the navigatins path
        category_nav = [
            (article_cats[i], article_cats[i + 1])
            for i in range(len(article_cats) - 1)
            if article_cats[i] in cat_df_clean["category"].values
            and article_cats[i + 1] in cat_df_clean["category"].values
        ]


        # Count pairwise category occurences
        cat_shift_counts = Counter(category_nav)

        # Convert to df
        cat_shift_df = pd.DataFrame(cat_shift_counts.items(), columns=["cat_move", "count"]) # type: ignore
        # Split cat_move col
        cat_shift_df[["from_cat", "to_cat"]] = pd.DataFrame(cat_shift_df["cat_move"].tolist(), index=cat_shift_df.index)
        # Drop cat_move col
        cat_shift_df = cat_shift_df.drop(columns="cat_move").sort_values("count", ascending=False).reset_index(drop=True)

        # Plot the most frequent categorical shifts
        cat_shift_df_filter = cat_shift_df.head(15)
        return cat_shift_df_filter

    def get_pairwise_category_shift_df(self, article_cats: pd.DataFrame, cat_df_clean: pd.DataFrame) -> pd.DataFrame:
        category_nav = [
            (article_cats[i], article_cats[i + 1])
            for i in range(len(article_cats) - 1)
            if article_cats[i] in cat_df_clean["category"].values
            and article_cats[i + 1] in cat_df_clean["category"].values
            and article_cats[i] != article_cats[i + 1]  # exclude same-category pairs
        ]

        # Count pairwise category occurences
        cat_shift_counts = Counter(category_nav)

        # Convert to df
        cat_shift_df = pd.DataFrame(cat_shift_counts.items(), columns=["cat_move", "count"]) # type: ignore
        # Split cat_move col
        cat_shift_df[["from_cat", "to_cat"]] = pd.DataFrame(cat_shift_df["cat_move"].tolist(), index=cat_shift_df.index)
        # Drop cat_move col
        cat_shift_df = cat_shift_df.drop(columns="cat_move").sort_values("count", ascending=False).reset_index(drop=True)

        # Plot the most frequent categorical shifts
        cat_shift_df_filter = cat_shift_df.head(15)
        return cat_shift_df_filter

    def get_category_shifts_df_pairs(self, article_cats_finished: pd.DataFrame, start_cat, end_cat: str) -> pd.DataFrame:
        # Buold the categorcal transitions along the navigation path for finished paths
        category_pairs = [
            # Category pairs encoding
            (f"{a}_{i}", f"{b}_{i+1}")
            # Ireate finished paths
            for cats in article_cats_finished  
            # Valid paths filter
            if len(cats) > 2 and cats[0] == start_cat and cats[-1] == end_cat 
            # Pairwise categories
            for i, (a, b) in enumerate(zip(cats[:-1], cats[1:]))  
            # Include or exclude missing categories (including chosen here to not loose granularity)
            #if a != "Unknown" and b != "Unknown"   
        ]

        # Create df of category frequencies along navigation paths from the pairs
        cat_shift_df_pairs = pd.DataFrame(Counter(category_pairs).items(), columns=["cat_move", "count"]) # type: ignore
        # Split cat_move col
        cat_shift_df_pairs[["cat_from", "cat_to"]] = pd.DataFrame(cat_shift_df_pairs["cat_move"].tolist(), index=cat_shift_df_pairs.index)
        # Drop cat_move col
        cat_shift_df_pairs.drop(columns="cat_move", inplace=True)
        return cat_shift_df_pairs

    def clean_llm_paths(self, df: pd.DataFrame, col):
        # Drop rows that have missing/empty path field
        df = df[df[col].notna() & (df[col] != "")].copy()
    
        # Parse the paths to python list
        def path_to_list(x):
            return ast.literal_eval(x) if isinstance(x, str) else None

        # Create col for converted lsit paths
        df["path_list"] = df[col].apply(path_to_list)

        # Remove rows with hallucinations ("none"/"None")
        df = df[~df["path_list"].apply(lambda path: any(str(p).lower() == "none" for p in path))]
    
        # Remove unrelated cols
        df = df[["id", "start", "destination", "path_list"]]
        return df

    def get_pairs_for_experiment(self,df: pd.DataFrame, num_pairs: int) -> pd.DataFrame:
        """
        Creates a balanced sample of start–destination pairs by difficulty:
            - Medium:      50% ≤ success rate < 75%
            - Hard:        25% ≤ success rate < 50%
            - Very Hard:   1%  ≤ success rate < 25%
            - Impossible:  success rate = 0%
        Selects the top (most-played) start–destination pairs from each bin.
        """

        # Aggregate success_count and sample_count per (start, destination)
        agg_df = df.groupby(["start", "destination"]).agg(
            success_count=("end", lambda x: x.notnull().sum()),
            sample_count=("end", "size")
        ).reset_index()

        # success rate
        agg_df["success_rate"] = agg_df["success_count"] / agg_df["sample_count"]

        # define difficulty bins
        bins = {
            "Medium": (0.50, 0.75),
            "Hard": (0.25, 0.50),
            # "Very Hard": (0.01, 0.25),
            # SKIP Impossible
            # "Impossible": (0.00, 0.00)  # exactly 0%
        }

        selected_pairs = []

        # select top pairs from each difficulty bin
        #result is a pd DataFrame with columns: start, destination, difficulty
        for difficulty, (lower, upper) in bins.items():
            # SKIP Impossible
            # if difficulty == "Impossible":
            #     bin_df = agg_df[agg_df["success_rate"] == 0.0]
            # else:
            #     bin_df = agg_df[(agg_df["success_rate"] >= lower) & (agg_df["success_rate"] < upper)]

            bin_df = agg_df[(agg_df["success_rate"] >= lower) & (agg_df["success_rate"] < upper)]

            # select top pairs by sample_count
            top_pairs = bin_df.nlargest(num_pairs, "sample_count")

            # annotate difficulty
            top_pairs = top_pairs.assign(difficulty=difficulty)

            selected_pairs.append(top_pairs)

        result = pd.concat(selected_pairs, ignore_index=True)

        # stack 6 times
        result = pd.concat([result] * 6, ignore_index=True)

        # add replicate index: 0, 1, 2
        result['replicate_idx'] = result.groupby(['start', 'destination']).cumcount()

        # global unique identifier
        result['id'] = (
            result['start']
            + "_"
            + result['destination']
            + "_"
            + result['replicate_idx'].astype(str)
        )
        return result


    # Compute number of categorical shifts per navigation path and the corresponding proportion
    def get_shifts_per_path(self, article_cats_finished):
        shifts_count = []
        shifts_prop = []

        for path in article_cats_finished:
            if len(path) < 2:
                continue
            shifts = sum(1 for a, b in zip(path[:-1], path[1:]) if a != b)
            shifts_count.append(shifts)
            shifts_prop.append(shifts / (len(path)-1))

        return shifts_count, shifts_prop

    # Compute average number of categorical shifts per navigation path
    def get_average_categorical_shifts_path(self, article_cats_finished):
        total_shifts = 0
        num_paths = 0

        for path in article_cats_finished:
            if len(path) < 2:
                continue
            shifts = sum(1 for a, b in zip(path[:-1], path[1:]) if a != b)
            total_shifts += shifts
            num_paths += 1

        if num_paths == 0:
            return 0.0

        return total_shifts / num_paths

    # Compute average number of categorical shifts per navigation step
    def get_average_categorical_shifts_step(self, article_cats_finished):
        total_shifts = 0
        total_pairs = 0

        for path in article_cats_finished:
            if len(path) < 2:
                continue
            shifts = sum(1 for a, b in zip(path[:-1], path[1:]) if a != b)
            total_shifts += shifts
            total_pairs += len(path) - 1

        if total_pairs == 0:
            return 0.0

        return total_shifts / total_pairs

    def compute_stepwise_shift_vectors(self, paths):
        step_vectors = []
    
        for path in paths:
            if len(path) < 2:
                continue
            shifts = [1 if a != b else 0 for a, b in zip(path[:-1], path[1:])]
            step_vectors.append(shifts)
    
        return step_vectors

    def compute_stepwise_mean_shifts(self, groups):
        mean_shifts = {}
    
        for name, step_vectors in groups.items():
            if len(step_vectors) == 0:
                continue
    
            resampled = []
    
            for v in step_vectors:
                if len(v) < 1:
                    continue
    
                x = np.linspace(0, 1, len(v))
                f = interp1d(x, v, kind="nearest", bounds_error=False, fill_value=np.nan)
                resampled.append(f(np.linspace(0, 1)))
    
            resampled = np.array(resampled)
            mean_shifts[name] = np.nanmean(resampled, axis=0)
    
        return mean_shifts

    def compute_highlevel_shifts(self, groups, article_to_highlevel):
        def path_to_highlevel(path):
            return [article_to_highlevel[a] for a in path if a in article_to_highlevel]

        # Convert paths to high-level categories
        highlevel_groups = {name: [path_to_highlevel(p) for p in paths] for name, paths in groups.items()}

        # Compute stepwise shifts for each group
        shifts_highlevel = {name: self.compute_stepwise_shift_vectors(paths)
                            for name, paths in highlevel_groups.items()}

        # Resample each group to normalized progress [0–1] and compute mean
        mean_shifts = {}
        for name, step_vectors in shifts_highlevel.items():
            if len(step_vectors) == 0:
                mean_shifts[name] = np.array([])
                continue

            # Determine max path length in this group for interpolation
            max_len = max(len(v) for v in step_vectors)
            progress_axis = np.linspace(0, 1, max_len)
            resampled_all = []

            for v in step_vectors:
                x = np.linspace(0, 1, len(v))
                f = interp1d(x, v, kind="nearest", bounds_error=False, fill_value=np.nan)
                resampled_all.append(f(progress_axis))

            resampled_all = np.array(resampled_all)
            mean_shifts[name] = np.nanmean(resampled_all, axis=0)

        return mean_shifts
    
    def get_human_llm_match_samples(self, df_human, llm_dfs, max_len=11, seed=42):
        random.seed(seed)
    
        # Get finished paths for humans and change seperator to comma
        human_paths = df_human[df_human["source"] == "finished"]["path"].str.split(";").tolist()
    
        # Combine LLM data on path_list col
        llm_paths = []
        for llm_df in llm_dfs:
            llm_paths.extend(llm_df['path_list'].tolist())
    
        # Get uique LLM start/dest pairs
        llm_pairs = set((p[0], p[-1]) for p in llm_paths)
    
        # Filter the human paths to only include LLM start/dest pairs and constrain max length to 11 (same as the prompts for LLM paths generation)
        human_paths_filt = [
            path for path in human_paths
            if (path[0], path[-1]) in llm_pairs and len(path) <= max_len
        ]
    
        # Sam balanced samples
        hum_sample_list = []
        llm_sample_list = []

        # Align and balance 
        for pair in llm_pairs:
            h_paths = [p for p in human_paths_filt if (p[0], p[-1]) == pair]
            l_paths = [p for p in llm_paths if (p[0], p[-1]) == pair]
            n_sample = min(len(h_paths), len(l_paths))
            if n_sample > 0:
                hum_sample_list.extend(random.sample(h_paths, n_sample))
                llm_sample_list.extend(random.sample(l_paths, n_sample))
    
        # Verify all pairs are balanced
        human_check = Counter((p[0], p[-1]) for p in hum_sample_list)
        llm_check = Counter((p[0], p[-1]) for p in llm_sample_list)
        mismatches = [(pair, human_check[pair], llm_check[pair])
                      for pair in human_check
                      if human_check[pair] != llm_check[pair]]
        
        # Check to ensure we dont have created biased data
        assert not mismatches, f"Mismatch found for pairs: {mismatches}"
        
        # Return balanced samples (current only hum_sample_list is used in main.ipynb)
        return hum_sample_list, llm_sample_list
        

    def compute_stepwise_mean_similarity(self, groups, article_to_idx, embeddings):
        def path_stepwise_similarity(path):
            sims = []
            for a, b in zip(path[:-1], path[1:]):
                if a in article_to_idx and b in article_to_idx:
                    emb_a = embeddings[article_to_idx[a]].reshape(1, -1)
                    emb_b = embeddings[article_to_idx[b]].reshape(1, -1)
                    sim = cosine_similarity(emb_a, emb_b)[0, 0]
                    sims.append(sim)
            return sims

        def path_stepwise_information_gain(path):
            if path is None or len(path) == 0:
                return []

            # ensure path is a list of strings
            path = [str(x) for x in path]

            start_title, dest_title = path[0], path[-1]
            if start_title not in article_to_idx or dest_title not in article_to_idx:
                return []

            emb_start = embeddings[article_to_idx[start_title]].reshape(1, -1)
            emb_dest  = embeddings[article_to_idx[dest_title]].reshape(1, -1)
            sims = []
            for curr_title in path[:-1]:
                if curr_title not in article_to_idx:
                    continue
                emb_curr = embeddings[article_to_idx[curr_title]].reshape(1, -1)
                sim_start = cosine_similarity(emb_curr, emb_start)[0, 0]
                sim_dest  = cosine_similarity(emb_curr, emb_dest)[0, 0]
                sims.append(sim_dest - sim_start)
            return sims
    
        # compute stepwise similarities for each player
        stepwise_similarity = {}
        stepwise_information_gains = {}
        for name, paths in groups.items():
            all_sims = [path_stepwise_similarity(p) for p in paths]
            stepwise_similarity[name] = all_sims

            all_information_gains = [path_stepwise_information_gain(p) for p in paths]
            stepwise_information_gains[name] = all_information_gains
    
        # Resample to same progress axis + compute mean similarity
        progress_axis = np.linspace(0, 1)
        mean_similarity = {}
    
        for name, sim_vectors in stepwise_similarity.items():
            resampled = []
            for v in sim_vectors:
                if len(v) < 1:
                    continue
                x = np.linspace(0, 1, len(v))
                f = interp1d(x, v, kind="nearest", bounds_error=False, fill_value=np.nan)
                resampled.append(f(progress_axis))
            resampled = np.array(resampled)
            mean_similarity[name] = np.nanmean(resampled, axis=0)

        mean_information_gains = {}
        for name, vectors in stepwise_information_gains.items():
            resampled = []
            for v in vectors:
                if len(v) < 1:
                    continue
                x = np.linspace(0, 1, len(v))
                f = interp1d(x, v, kind="nearest", bounds_error=False, fill_value=np.nan)
                resampled.append(f(progress_axis))
            resampled = np.array(resampled)
            mean_information_gains[name] = np.nanmean(resampled, axis=0)
    
        return mean_similarity, progress_axis, stepwise_similarity, mean_information_gains


    def compute_flat_stepwise_similarity(self, groups, article_to_idx, embeddings):
        stepwise_similarity_flat = {}
        for name, paths in groups.items():
            all_sims = []
            for path in paths:
                for a, b in zip(path[:-1], path[1:]):
                    if a in article_to_idx and b in article_to_idx:
                        emb_a = embeddings[article_to_idx[a]].reshape(1, -1)
                        emb_b = embeddings[article_to_idx[b]].reshape(1, -1)
                        sim = cosine_similarity(emb_a, emb_b)[0, 0]
                        sim = np.clip(sim, 0, 1)
                        all_sims.append(sim)
            stepwise_similarity_flat[name] = all_sims

        return stepwise_similarity_flat

    def resolve_llm_path(self,llm_path:pd.DataFrame):
        llm_df = llm_path.copy()
        def path_to_list(x):
            return ast.literal_eval(x) if isinstance(x, str) else None

        def is_valid_path(path, dest):
            return path is not None and len(path) > 0 and path[-1] == dest
        llm_path_stats = llm_df.copy()
        llm_path_stats['link_aware_paths'] = llm_path_stats['link_aware_paths'].apply(path_to_list)
        llm_path_stats['blind_paths'] = llm_path_stats['blind_paths'].apply(path_to_list)
        llm_path_stats['external_info_paths'] = llm_path_stats['external_info_paths'].apply(path_to_list)
        llm_path_stats['tot_paths'] = llm_path_stats['tot_paths'].apply(path_to_list)
        llm_path_stats['link_aware_success'] = llm_path_stats.apply(
            lambda row: is_valid_path(row['link_aware_paths'], row['destination']), axis=1)
        llm_path_stats['blind_success'] = llm_path_stats.apply(
            lambda row: is_valid_path(row['blind_paths'], row['destination']), axis=1)
        llm_path_stats['external_info_success'] = llm_path_stats.apply(
            lambda row: is_valid_path(row['external_info_paths'], row['destination']), axis=1)
        llm_path_stats['tot_success'] = llm_path_stats.apply(
            lambda row: is_valid_path(row['tot_paths'], row['destination']), axis=1)
        return llm_path_stats