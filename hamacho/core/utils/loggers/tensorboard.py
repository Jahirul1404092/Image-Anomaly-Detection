"""Tensorboard logger with add image interface."""


from typing import Any
from typing import Optional
from typing import Union

import numpy as np
from matplotlib.figure import Figure
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from .base import ImageLoggerBase


class TensorBoardLogger(ImageLoggerBase, TensorBoardLogger):
    """Logger for tensorboard.

    Adds interface for `add_image` in the logger rather than calling the experiment object.

    Note:
        Same as the Tensorboard Logger provided by PyTorch Lightning and the doc string is reproduced below.

    Logs are saved to
    ``os.path.join(save_dir, name, version)``. This is the default logger in Lightning, it comes
    preinstalled.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from hamacho.core.utils.loggers import TensorBoardLogger
        >>> logger = TensorBoardLogger("tb_logs", name="my_model")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir (str): Save directory
        name (Optional, str): Experiment name. Defaults to ``'default'``. If it is the empty string then no
            per-experiment subdirectory is used.
        version (Optional, int, str): Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory name,
            otherwise ``'version_${version}'`` is used.
        log_graph (bool): Adds the computational graph to tensorboard. This requires that
            the user has defined the `self.example_input_array` attribute in their
            model.
        default_hp_metric (bool): Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is
            called without a metric (otherwise calls to log_hyperparams without a metric are ignored).
        prefix (str): A string to put at the beginning of metric keys.
        **kwargs: Additional arguments like `comment`, `filename_suffix`, etc. used by
            :class:`SummaryWriter` can be passed as keyword arguments in this logger.
    """

    def __init__(
        self,
        save_dir: str,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        **kwargs
    ):
        super().__init__(
            save_dir,
            name=name,
            version=version,
            log_graph=log_graph,
            default_hp_metric=default_hp_metric,
            prefix=prefix,
            **kwargs
        )

    @rank_zero_only
    def add_image(
        self,
        image: Union[np.ndarray, Figure],
        name: Optional[str] = None,
        **kwargs: Any
    ):
        """Interface to add image to tensorboard logger.

        Args:
            image (Union[np.ndarray, Figure]): Image to log
            name (Optional[str]): The tag of the image
            kwargs: Accepts only `global_step` (int). The step at which to log the image.
        """
        if "global_step" not in kwargs:
            raise ValueError("`global_step` is required for tensorboard logger")

        # Matplotlib Figure is not supported by tensorboard
        if isinstance(image, Figure):
            axis = image.gca()
            axis.axis("off")
            axis.margins(0)
            image.canvas.draw()  # cache the renderer
            buffer = np.frombuffer(image.canvas.tostring_rgb(), dtype=np.uint8)
            image = buffer.reshape(image.canvas.get_width_height()[::-1] + (3,))
            kwargs["dataformats"] = "HWC"

        self.experiment.add_image(img_tensor=image, tag=name, **kwargs)
