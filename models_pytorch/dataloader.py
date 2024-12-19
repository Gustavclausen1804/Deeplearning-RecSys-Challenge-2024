from dataclasses import dataclass, field
import time
import torch
from torch.utils.data import Dataset, DataLoader
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
    """
    A PyTorch Dataset for news recommendation.
    """
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
    test: bool = False

    def __post_init__(self):
        """
        Post-initialization method. Loads the data and sets additional attributes.
        """
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        if self.test:
            self.X = self.load_data()
        else:
            self.X, self.y = self.load_data()
        if self.kwargs:
            self.set_kwargs(self.kwargs)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_data(self) -> tuple[pl.DataFrame, pl.Series]:
        X = self.behaviors.with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        if self.test:
            return X
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
    def __post_init__(self):
        # Initialize parent class attributes first
        super().__post_init__()
        
        print("Starting preprocessing...")
        # Now preprocess the data
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
        
        # Transform the entire dataset once
        self.X = self.transform(self.X)
        
        # Preprocess all samples
        self.samples = []
        print(self.X.shape)
        for idx in range(len(self.X)):
            sample_X = self.X[idx]
            if not self.test:
                sample_y = self.y[idx]
            
            # Extract lists
            history_list = sample_X[self.history_column].to_list()[0]  # Added [0]
            inview_list = sample_X[self.inview_col].to_list()[0]      # Added [0]
            impression_id = sample_X['impression_id'].to_list()[0]     # Added [0]
            
            # Map IDs to embeddings
            his_input_title = self.lookup_article_matrix[history_list]
            pred_input_title = self.lookup_article_matrix[inview_list]
            
            # Convert to tensors
            his_input_title = torch.tensor(his_input_title, dtype=torch.float32)
            if his_input_title.ndim > 2:
                his_input_title = np.squeeze(his_input_title, axis=1)
            pred_input_title = torch.tensor(pred_input_title, dtype=torch.float32)
            if pred_input_title.ndim > 2:
                pred_input_title = np.squeeze(pred_input_title, axis=1)
            if not self.test:
                sample_y_tensor = torch.tensor(sample_y.to_list(), dtype=torch.float32)  # Added to_list()
            impression_id_tensor = torch.tensor(impression_id, dtype=torch.int64)
            
            if not self.test:
                self.samples.append((
                (his_input_title, pred_input_title),
                sample_y_tensor,
                impression_id_tensor
                ))
            else:
                self.samples.append((
                (his_input_title, pred_input_title),
                impression_id_tensor
                ))

        end_time = time.time()
        print(f"Data preprocessing completed in {end_time - start_time:.2f} seconds.")

    def __getitem__(self, idx):
        return self.samples[idx]