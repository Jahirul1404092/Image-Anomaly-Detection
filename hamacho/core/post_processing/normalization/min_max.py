"""Tools for min-max normalization."""


from typing import Union

import numpy as np
import torch
from torch import Tensor


def normalize(
    targets: Union[np.ndarray, Tensor, np.float32],
    threshold: Union[np.ndarray, Tensor, float],
    min_val: Union[np.ndarray, Tensor, float],
    max_val: Union[np.ndarray, Tensor, float],
) -> Union[np.ndarray, Tensor]:
    """Apply min-max normalization and shift the values such that the threshold value is centered at 0.5."""
    normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
    if isinstance(targets, (np.ndarray, np.float32)):
        normalized = np.minimum(normalized, 1)
        normalized = np.maximum(normalized, 0)
    elif isinstance(targets, Tensor):
        normalized = torch.minimum(
            normalized, torch.tensor(1)
        )  # pylint: disable=not-callable
        normalized = torch.maximum(
            normalized, torch.tensor(0)
        )  # pylint: disable=not-callable
    else:
        raise ValueError(
            f"Targets must be either Tensor or Numpy array. Received {type(targets)}"
        )
    return normalized
