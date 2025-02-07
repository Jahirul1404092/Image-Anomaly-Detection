"""Utilities to help tests inferencers"""




from typing import Dict, Iterable, List, Tuple

import numpy as np

from hamacho.plug_in.models.components import AnomalyModule


class MockImageLoader:
    """Create mock images for inference on CPU based on the specifics of the original torch test dataset.
    Uses yield so as to avoid storing everything in the memory.
    Args:
        image_size (List[int]): Size of input image
        total_count (int): Total images in the test dataset
    """

    def __init__(self, image_size: List[int], total_count: int):
        self.total_count = total_count
        self.image_size = image_size
        self.image = np.ones((*self.image_size, 3)).astype(np.uint8)

    def __len__(self):
        """Get total count of images."""
        return self.total_count

    def __call__(self) -> Iterable[np.ndarray]:
        """Yield batch of generated images.
        Args:
            idx (int): Unused
        """
        for _ in range(self.total_count):
            yield self.image


def get_meta_data(model: AnomalyModule, input_size: Tuple[int, int]) -> Dict:
    """Get meta data for inference.
    Args:
        model (AnomalyModule): Trained model from which the metadata is extracted.
        input_size (Tuple[int, int]): Input size used to resize the pixel level mean and std.
    Returns:
        (Dict): Metadata as dictionary.
    """
    meta_data = {
        "image_threshold": model.image_threshold.value.cpu().numpy(),
        "pixel_threshold": model.pixel_threshold.value.cpu().numpy(),
        "stats": {},
    }

    image_mean = model.training_distribution.image_mean.cpu().numpy()
    if image_mean.size > 0:
        meta_data["stats"]["image_mean"] = image_mean

    image_std = model.training_distribution.image_std.cpu().numpy()
    if image_std.size > 0:
        meta_data["stats"]["image_std"] = image_std

    pixel_mean = model.training_distribution.pixel_mean.cpu().numpy()
    if pixel_mean.size > 0:
        meta_data["stats"]["pixel_mean"] = pixel_mean.reshape(input_size)

    pixel_std = model.training_distribution.pixel_std.cpu().numpy()
    if pixel_std.size > 0:
        meta_data["stats"]["pixel_std"] = pixel_std.reshape(input_size)

    return meta_data
