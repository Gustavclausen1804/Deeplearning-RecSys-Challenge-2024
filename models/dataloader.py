from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from typing import Tuple, Dict, Any
import os
from utils._articles_behaviors import map_list_article_id_to_value
from utils._python import (
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)
from utils._constants import (
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_USER_COL,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class NewsRecDataset(Dataset):
    """Base dataset class for news recommendation"""
    behaviors: pl.DataFrame
    history_column: str
    article_dict: Dict[int, Any]
    unknown_representation: str
    eval_mode: bool = False
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    labels_col: str = DEFAULT_LABELS_COL
    user_col: str = DEFAULT_USER_COL
    kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize lookup tables and load data"""
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        if self.kwargs is not None:
            for key, value in self.kwargs.items():
                setattr(self, key, value)
        
        # Convert lookup matrix to torch tensor
        self.lookup_article_matrix = torch.tensor(
            self.lookup_article_matrix, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.X)

    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y

    def set_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)


class NRMSDataset(NewsRecDataset):
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get a single sample from the dataset"""
        row_X = self.X[idx:idx+1]
        row_y = self.y[idx:idx+1]
        
        # Transform the data
        batch_X = self.transform(row_X)
        
        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            batch_y = torch.tensor(
                np.array(row_y.explode().to_list()).reshape(-1, 1),
                dtype=torch.float32
            )
            
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix.numpy(),
                repeats=repeats,
            )
            
            pred_input_title = self.lookup_article_matrix[
                torch.tensor(batch_X[self.inview_col].explode().to_list())
            ]
        else:
            batch_y = torch.tensor(np.array(row_y.to_list()), dtype=torch.float32)
            his_input_title = self.lookup_article_matrix[
                torch.tensor(batch_X[self.history_column].to_list())
            ]
            pred_input_title = self.lookup_article_matrix[
                torch.tensor(batch_X[self.inview_col].to_list())
            ]
            pred_input_title = torch.squeeze(pred_input_title, dim=2)

        his_input_title = torch.squeeze(his_input_title, dim=2)
        
        return (his_input_title, pred_input_title), batch_y


def create_nrms_dataloaders(
    train_behaviors: pl.DataFrame,
    val_behaviors: pl.DataFrame,
    article_dict: Dict[int, Any],
    history_column: str,
    train_batch_size: int = 64,
    val_batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    train_dataset = NRMSDataset(
        behaviors=train_behaviors,
        article_dict=article_dict,
        unknown_representation="zeros",
        history_column=history_column,
        eval_mode=False,
        **kwargs
    )
    
    val_dataset = NRMSDataset(
        behaviors=val_behaviors,
        article_dict=article_dict,
        unknown_representation="zeros",
        history_column=history_column,
        eval_mode=True,
        **kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

@dataclass
class NRMSDataLoaderPretransform(NewsRecDataset):
    """
    In the __post_init__ pre-transform the entire DataFrame. This is useful for
    when data can fit in memory, as it will be much faster ones training.
    Note, it might not be as scaleable.
    """

    def __post_init__(self):
        super().__post_init__()
        self.X = self.X.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        his_input_title:    (samples, history_size, document_dimension)
        pred_input_title:   (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # =>
        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            # =>
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            # =>
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            # =>
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
        else:
            batch_y = np.array(batch_y.to_list())
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)

        his_input_title = np.squeeze(his_input_title, axis=2)
        return (his_input_title, pred_input_title), batch_y
