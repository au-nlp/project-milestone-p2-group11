import pandas as pd
import numpy as np
import csv
import logging
from collections import defaultdict, Counter

from sympy.integrals.meijerint_doc import category

from config_local import Config
import urllib.parse

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

        return pd.read_csv(path, comment='#', sep='\t', names=column_names)

    def get_merged_input_files(self) -> pd.DataFrame:
        # parsing input files into pd.DataFrames
        parsed_dfs = []
        for input_file in self.config.input_files_paths:
            df = self.parse_input_file(input_file.path)
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
        finished_paths: pd.DataFrame = agg_with_start_and_end[agg_with_start_and_end['source'] == 'finished'] # type: ignore
        unfinished_paths: pd.DataFrame = agg_with_start_and_end[agg_with_start_and_end['source'] == 'unfinished'] # type: ignore
        
        # merge start - destination on top k start paths
        top_finished_paths = pd.merge(
            finished_paths,
            target_groups_df[['start', 'source']],
            on=['start', 'source'],
            how='inner'
        )
        
        # filter out top 3 destinations for each of the start paths
        top_3_finished_dest = top_finished_paths.sort_values('sample_count', ascending=False).groupby('start').head(3)
        
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
        return pd.concat([top_3_finished_dest, matching_unfinished_paths]).sort_values(['start', 'source', 'destination'])

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

    def load_data(self):
        articles = pd.read_csv(self.config.input_file_articles, sep='\t', skiprows=12, names=['article'])
        categories = pd.read_csv(self.config.input_file_categories, sep='\t', skiprows=12, names=['article', 'category'])
        links = pd.read_csv(self.config.input_file_links, sep='\t', skiprows=11, names=['linkSource', 'linkTarget'])
        articles = articles.map(urllib.parse.unquote)
        categories = categories.map(urllib.parse.unquote)
        links = links.map(urllib.parse.unquote)
        return articles, categories, links

    def resolve_all_parsed_paths(self, df: pd.DataFrame) -> None:
        if(type(df['path'].iloc[0]) == list):
            return
        df['path'] = df['path'].str.split(';').apply(self._resolve_back_clicks)
        df['num_hops'] = df['path'].str.len()