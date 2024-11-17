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

@dataclass
class NewsRecDataset(Dataset):
    behaviors: pl.DataFrame
    article_dict: Dict[int, Any]
    history_column: str
    unknown_representation: str
    eval_mode: bool = False
    batch_size: int = field(default=32)
    inview_col: str = field(default=DEFAULT_INVIEW_ARTICLES_COL)
    labels_col: str = field(default=DEFAULT_LABELS_COL)
    user_col: str = field(default=DEFAULT_USER_COL)
    kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        print(f"Loaded data: X shape = {len(self.X)}, y shape = {len(self.y)}")
        if self.kwargs is not None:
            self.set_kwargs(self.kwargs)
        
        self.lookup_article_matrix = torch.tensor(
            self.lookup_article_matrix, dtype=torch.float32
        )

    def __len__(self) -> int:
        length = int(np.ceil(len(self.X) / float(self.batch_size)))
        print(f"Dataset length requested: {length}")
        return length

    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y
        
    def set_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)

@dataclass
class NRMSDataset(NewsRecDataset):
    def __post_init__(self):
        super().__post_init__()

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

    def __getitem__(self, idx) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        print("WTF------------------")
        print(f"\nPyTorch Dataloader - Batch {idx}:")
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        print("Initial batch_X shape:", len(batch_X))
        print("Initial batch_y shape:", len(batch_y))

        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            batch_y = torch.tensor(
                np.array(batch_y.explode().to_list()).reshape(-1, 1),
                dtype=torch.float32
            )
            
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix.numpy(),
                repeats=repeats,
            )
            his_input_title = torch.tensor(his_input_title)
            
            pred_input_title = self.lookup_article_matrix[
                torch.tensor(batch_X[self.inview_col].explode().to_list())
            ]
        else:
            batch_y = torch.tensor(np.array(batch_y.to_list()), dtype=torch.float32)
            print("Training batch_y shape:", batch_y.shape)
            
            his_input_title = self.lookup_article_matrix[
                torch.tensor(batch_X[self.history_column].to_list(), dtype=torch.long)
            ].unsqueeze(2)
            print("his_input_title initial shape:", his_input_title.shape)
            
            pred_input_title = self.lookup_article_matrix[
                torch.tensor(batch_X[self.inview_col].to_list(), dtype=torch.long)
            ].unsqueeze(2)
            print("pred_input_title initial shape:", pred_input_title.shape)

            # Make sure to match TF shapes exactly
            his_input_title = his_input_title.squeeze(2)
            pred_input_title = pred_input_title.squeeze(2)

        print("\nFinal shapes:")
        print("his_input_title:", his_input_title.shape)
        print("pred_input_title:", pred_input_title.shape)
        print("batch_y:", batch_y.shape)
        
        print("\nSample values:")
        print("his_input_title first element:", his_input_title[0, 0, :5].tolist())
        print("pred_input_title first element:", pred_input_title[0, 0, :5].tolist())
        print("batch_y first element:", batch_y[0].tolist())

        return (his_input_title, pred_input_title), batch_y

def create_nrms_dataloaders(
    train_behaviors: pl.DataFrame,
    val_behaviors: pl.DataFrame,
    article_dict: Dict[int, Any],
    history_column: str,
    train_batch_size: int = 64,
    val_batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = NRMSDataset(
        behaviors=train_behaviors,
        article_dict=article_dict,
        unknown_representation="zeros",
        history_column=history_column,
        eval_mode=False,
        batch_size=train_batch_size
    )
    
    val_dataset = NRMSDataset(
        behaviors=val_behaviors,
        article_dict=article_dict,
        unknown_representation="zeros",
        history_column=history_column,
        eval_mode=True,
        batch_size=val_batch_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

@dataclass
class NRMSDatasetPretransform(NewsRecDataset):
    """Equivalent to TF NRMSDataLoaderPretransform"""
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

    def __getitem__(self, idx) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Same implementation as NRMSDataset but without transform call"""
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            batch_y = torch.tensor(
                np.array(batch_y.explode().to_list()).reshape(-1, 1),
                dtype=torch.float32
            )
            
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix.numpy(),
                repeats=repeats,
            )
            his_input_title = torch.tensor(his_input_title)
            
            pred_input_title = self.lookup_article_matrix[
                torch.tensor(batch_X[self.inview_col].explode().to_list())
            ]
        else:
            batch_y = torch.tensor(np.array(batch_y.to_list()), dtype=torch.float32)
            his_input_title = self.lookup_article_matrix[
                torch.tensor(batch_X[self.history_column].to_list())
            ]
            pred_input_title = self.lookup_article_matrix[
                torch.tensor(batch_X[self.inview_col].to_list())
            ]
            pred_input_title = torch.squeeze(pred_input_title, dim=2)

        his_input_title = torch.squeeze(his_input_title, dim=2)
        return (his_input_title, pred_input_title), batch_y
