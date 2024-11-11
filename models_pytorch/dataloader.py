from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np

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

    def __post_init__(self):
        """
        Post-initialization method. Loads the data and sets additional attributes.
        """
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
class NRMSDataLoader(NewsrecDataLoader):
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

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Returns:
            Inputs: (his_input_title, pred_input_title)
            Targets: batch_y
        """
        batch_X = self.X[idx].pipe(self.transform)
        batch_y = self.y[idx]

        if self.eval_mode:
            repeats = batch_X["n_samples"]
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
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

        # Convert to torch tensors
        his_input_title = torch.tensor(his_input_title, dtype=torch.float32)
        pred_input_title = torch.tensor(pred_input_title, dtype=torch.float32)
        batch_y = torch.tensor(batch_y, dtype=torch.float32)

        return (his_input_title, pred_input_title), batch_y
