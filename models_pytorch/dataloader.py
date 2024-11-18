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

    def __getitem__(self, idx):
        try:
            sample_X = self.X[idx].pipe(self.transform)
            sample_y = self.y[idx]

            # Extract lists
            history_list = sample_X[self.history_column].to_list()[0]
            inview_list = sample_X[self.inview_col].to_list()[0]

            # Check for empty lists
            if not history_list:
                print(f"Empty history_list at index {idx}")
                raise ValueError(f"Empty history_list at index {idx}")
            if not inview_list:
                print(f"Empty inview_list at index {idx}")
                raise ValueError(f"Empty inview_list at index {idx}")

            # Map IDs to embeddings
            his_input_title = self.lookup_article_matrix[history_list]
            pred_input_title = self.lookup_article_matrix[inview_list]

            # Squeeze singleton dimensions if necessary
            if his_input_title.ndim > 2:
                his_input_title = np.squeeze(his_input_title, axis=1)
            if pred_input_title.ndim > 2:
                pred_input_title = np.squeeze(pred_input_title, axis=1)

            # Convert to tensors
            his_input_title = torch.tensor(his_input_title, dtype=torch.float32)
            pred_input_title = torch.tensor(pred_input_title, dtype=torch.float32)

            # Process sample_y correctly
            sample_y_list = sample_y.to_list()
            if not sample_y_list:
                print(f"Empty sample_y at index {idx}")
                raise ValueError(f"Empty sample_y at index {idx}")

            sample_y_tensor = torch.tensor(sample_y_list, dtype=torch.float32)

            return (his_input_title, pred_input_title), sample_y_tensor
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            raise


