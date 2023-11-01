"""Image Utils."""



import math
from pathlib import Path
from typing import List
from typing import Union

import cv2
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torchvision.datasets.folder import IMG_EXTENSIONS


def get_image_filenames(path: Union[str, Path]) -> List[str]:
    """Get image filenames.

    Args:
        path (Union[str, Path]): Path to image or image-folder.

    Returns:
        List[str]: List of image filenames

    """
    image_filenames: List[str]

    if isinstance(path, str):
        path = Path(path)

    if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
        image_filenames = [str(path)]

    if path.is_dir():
        image_filenames = [
            str(p) for p in path.glob("**/*") if p.suffix.lower() in IMG_EXTENSIONS
        ]

    if len(image_filenames) == 0:
        raise ValueError(f"Found 0 images in {path}")

    return image_filenames


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Read image from disk in RGB format.

    Args:
        path (str, Path): path to the image file

    Example:
        >>> image = read_image("test_image.jpg")

    Returns:
        image as numpy array
    """
    path = path if isinstance(path, str) else str(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def pad_nextpow2(batch: Tensor) -> Tensor:
    """Compute required padding from input size and return padded images.

    Finds the largest dimension and computes a square image of dimensions that are of the power of 2.
    In case the image dimension is odd, it returns the image with an extra padding on one side.

    Args:
        batch (Tensor): Input images

    Returns:
        batch: Padded batch
    """
    # find the largest dimension
    l_dim = 2 ** math.ceil(math.log(max(*batch.shape[-2:]), 2))
    padding_w = [
        math.ceil((l_dim - batch.shape[-2]) / 2),
        math.floor((l_dim - batch.shape[-2]) / 2),
    ]
    padding_h = [
        math.ceil((l_dim - batch.shape[-1]) / 2),
        math.floor((l_dim - batch.shape[-1]) / 2),
    ]
    padded_batch = F.pad(batch, pad=[*padding_h, *padding_w])
    return padded_batch


def add_label(
    prediction: np.ndarray, scores: float, font: int = cv2.FONT_HERSHEY_PLAIN
) -> np.ndarray:
    """If the model outputs score, it adds the score to the output image.
    Args:
        prediction (np.ndarray): Resized anomaly map.
        scores (float): Confidence score.
    Returns:
        np.ndarray: Image with score text.
    """
    text = f"Conf. Score {scores:.0%}"
    font_size = (
        prediction.shape[1] // 1024 + 1
    )  # Text scale is calculated based on the reference size of 1024
    (width, height), baseline = cv2.getTextSize(text, font, font_size, thickness=0)
    label_patch = np.zeros((height + baseline, width + baseline, 3), dtype=np.uint8)
    label_patch[:, :] = (225, 252, 134)
    cv2.putText(
        label_patch,
        text,
        (0, baseline // 2 + height),
        font,
        font_size,
        0,
        lineType=cv2.LINE_AA,
    )
    if (
        label_patch.shape[1]
        > prediction[: baseline + height, : baseline + width].shape[1]
    ):
        fx = (label_patch.shape[1] + 1) / prediction.shape[1]
        prediction = cv2.resize(prediction, dsize=None, fx=fx, fy=fx)

    prediction[: baseline + height, : baseline + width] = label_patch
    return prediction
