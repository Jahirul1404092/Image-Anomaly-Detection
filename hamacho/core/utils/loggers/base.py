"""Base logger for image logging consistency across all loggers used in hamacho."""


from abc import abstractmethod
from typing import Any
from typing import Optional
from typing import Union

import numpy as np
from matplotlib.figure import Figure


class ImageLoggerBase:
    """Adds a common interface for logging the images."""

    @abstractmethod
    def add_image(
        self,
        image: Union[np.ndarray, Figure],
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Interface to log images in the respective loggers."""
        raise NotImplementedError()
