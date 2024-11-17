from collections import Counter
from typing import Iterable
import numpy as np

def convert_to_binary(y_pred: np.ndarray, threshold: float):
    y_pred = np.asarray(y_pred)
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0
    return y_pred