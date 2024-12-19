from dataclasses import dataclass, field
import time
import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np

from utils._articles_behaviors import map_list_article_id_to_value_optimized
from utils._python import (
    generate_unique_name,
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)
from utils._constants import (
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_USER_COL,
)


@dataclass
class NewsrecDataLoader(Dataset):
    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, any]
    unknown_representation: str
    eval_mode: bool = False
    batch_size: int = 32
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    labels_col: str = DEFAULT_LABELS_COL
    user_col: str = DEFAULT_USER_COL
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        if self.kwargs:
            self.set_kwargs(self.kwargs)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_data(self) -> tuple[pl.DataFrame, pl.Series]:
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y

    def set_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class NRMSDataSet(NewsrecDataLoader):
    category_mapping: dict[int, np.ndarray] = field(default_factory=dict)
    topic_mapping: dict[int, np.ndarray] = field(default_factory=dict)
    sentiment_mapping: dict[int, float] = field(default_factory=dict)
    read_time_mapping: dict[int, float] = field(default_factory=dict)
    pageviews_mapping: dict[int, float] = field(default_factory=dict)
    timestamp_mapping: dict[int, float] = field(default_factory=dict)

    category_emb_dim: int = 128
    topic_emb_dim: int = 128

    def __post_init__(self):
        super().__post_init__()
        print("Starting preprocessing...")
        self.preprocess_data()

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        BATCH_SIZE = 10000
        return df.pipe(
            map_list_article_id_to_value_optimized,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
            batch_size=BATCH_SIZE
        ).pipe(
            map_list_article_id_to_value_optimized,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
            batch_size=BATCH_SIZE
        )

    def preprocess_data(self):
        print("Preprocessing data...")
        start_time = time.time()

        self.X = self.transform(self.X)

        # Default values
        default_category_emb = np.zeros(self.category_emb_dim, dtype=np.float32)
        default_topic_emb = np.zeros(self.topic_emb_dim, dtype=np.float32)
        default_sentiment = 0.0
        default_read_time = 0.0
        default_pageviews = 0.0

        self.samples = []
        print(self.X.shape)
        for idx in range(len(self.X)):
            sample_X = self.X[idx]
            sample_y = self.y[idx]

            # Extract lists
            history_list = sample_X[self.history_column].to_list()[0]
            inview_list = sample_X[self.inview_col].to_list()[0]

            # Flatten if nested
            if len(history_list) > 0 and isinstance(history_list[0], list):
                history_list = [x[0] for x in history_list]

            if len(inview_list) > 0 and isinstance(inview_list[0], list):
                inview_list = [x[0] for x in inview_list]

            history_list = [int(x) for x in history_list]
            inview_list = [int(x) for x in inview_list]

            impression_id = sample_X['impression_id'].to_list()[0]
            impression_timestamp = sample_X['impression_time'].to_list()[0]
            

            # Map IDs to article embeddings
            his_input_title = self.lookup_article_matrix[history_list]
            pred_input_title = self.lookup_article_matrix[inview_list]

            # For user history, map IDs to features
            his_category_emb_list = [self.category_mapping.get(art_id, default_category_emb) for art_id in history_list]
            his_topic_emb_list = [self.topic_mapping.get(art_id, default_topic_emb) for art_id in history_list]
            his_sentiment_scores_list = [self.sentiment_mapping.get(art_id, default_sentiment) for art_id in history_list]
            his_read_times_list = [self.read_time_mapping.get(art_id, default_read_time) for art_id in history_list]
            his_pageviews_list = [self.pageviews_mapping.get(art_id, default_pageviews) for art_id in history_list]
            his_timestamps_list = [self.timestamp_mapping.get(art_id, 0.0) for art_id in history_list]

            his_category_emb = np.array(his_category_emb_list, dtype=np.float32)
            his_topic_emb = np.array(his_topic_emb_list, dtype=np.float32)
            his_sentiment_scores = np.array(his_sentiment_scores_list, dtype=np.float32)
            his_read_times = np.array(his_read_times_list, dtype=np.float32)
            his_pageviews = np.array(his_pageviews_list, dtype=np.float32)
            his_timestamps = np.array(his_timestamps_list, dtype=np.float32)

            # For candidate articles
            pred_category_emb_list = [self.category_mapping.get(art_id, default_category_emb) for art_id in inview_list]
            pred_topic_emb_list = [self.topic_mapping.get(art_id, default_topic_emb) for art_id in inview_list]
            pred_sentiment_scores_list = [self.sentiment_mapping.get(art_id, default_sentiment) for art_id in inview_list]
            pred_read_times_list = [self.read_time_mapping.get(art_id, default_read_time) for art_id in inview_list]
            pred_pageviews_list = [self.pageviews_mapping.get(art_id, default_pageviews) for art_id in inview_list]
            pred_timestamps_list = [self.timestamp_mapping.get(art_id, 0.0) for art_id in inview_list]

            pred_category_emb = np.array(pred_category_emb_list, dtype=np.float32)
            pred_topic_emb = np.array(pred_topic_emb_list, dtype=np.float32)
            pred_sentiment_scores = np.array(pred_sentiment_scores_list, dtype=np.float32)
            pred_read_times = np.array(pred_read_times_list, dtype=np.float32)
            pred_pageviews = np.array(pred_pageviews_list, dtype=np.float32)
            pred_timestamps = np.array(pred_timestamps_list, dtype=np.float32)
            

            # Convert all to tensors
            his_input_title = torch.tensor(his_input_title, dtype=torch.float32)
            if his_input_title.ndim > 2:
                his_input_title = np.squeeze(his_input_title, axis=1)
                his_input_title = torch.tensor(his_input_title, dtype=torch.float32)

            pred_input_title = torch.tensor(pred_input_title, dtype=torch.float32)
            if pred_input_title.ndim > 2:
                pred_input_title = np.squeeze(pred_input_title, axis=1)
                pred_input_title = torch.tensor(pred_input_title, dtype=torch.float32)

            his_category_emb = torch.tensor(his_category_emb, dtype=torch.float32)
            his_topic_emb = torch.tensor(his_topic_emb, dtype=torch.float32)
            his_sentiment_scores = torch.tensor(his_sentiment_scores, dtype=torch.float32)
            his_read_times = torch.tensor(his_read_times, dtype=torch.float32)
            his_pageviews = torch.tensor(his_pageviews, dtype=torch.float32)
            his_timestamps = torch.tensor(his_timestamps, dtype=torch.float32)

            pred_category_emb = torch.tensor(pred_category_emb, dtype=torch.float32)
            pred_topic_emb = torch.tensor(pred_topic_emb, dtype=torch.float32)
            pred_sentiment_scores = torch.tensor(pred_sentiment_scores, dtype=torch.float32)
            pred_read_times = torch.tensor(pred_read_times, dtype=torch.float32)
            pred_pageviews = torch.tensor(pred_pageviews, dtype=torch.float32)

            sample_y_tensor = torch.tensor(sample_y.to_list(), dtype=torch.float32)
            impression_id_tensor = torch.tensor(impression_id, dtype=torch.int64)
            impression_timestamp_tensor = torch.tensor(impression_timestamp, dtype=torch.int64)
            pred_timestamps = torch.tensor(pred_timestamps, dtype=torch.float32)

            # Final sample structure:
            # ((his_input_title, his_category_emb, his_topic_emb, his_sentiment_scores, his_read_times, his_pageviews,
            #   pred_input_title, pred_category_emb, pred_topic_emb, pred_sentiment_scores, pred_read_times, pred_pageviews),
            #  sample_y_tensor, impression_id_tensor)
            self.samples.append((
                (his_input_title, his_category_emb, his_topic_emb, his_sentiment_scores, his_read_times, his_pageviews, his_timestamps,
                 pred_input_title, pred_category_emb, pred_topic_emb, pred_sentiment_scores, pred_read_times, pred_pageviews, pred_timestamps, impression_timestamp_tensor),
                sample_y_tensor,
                impression_id_tensor
            ))

        end_time = time.time()
        print(f"Data preprocessing completed in {end_time - start_time:.2f} seconds.")

    def __getitem__(self, idx):
        return self.samples[idx]
