"""Callback that compresses a trained model by first exporting to .onnx format, and then converting to OpenVINO IR."""


import logging
import os
from typing import Tuple

from pytorch_lightning import Callback

from hamacho.core.deploy import export_convert
from hamacho.plug_in.models.components import AnomalyModule


logger = logging.getLogger(__name__)


class OpenVINOCallback(Callback):
    """Callback to compresses a trained model.

    Model is first exported to ``.onnx`` format, and then converted to OpenVINO IR.

    Args:
        input_size (Tuple[int, int]): Tuple of image height, width
        dirpath (str): Path for model output
        filename (str): Name of output model
    """

    def __init__(self, input_size: Tuple[int, int], dirpath: str, filename: str):
        self.input_size = input_size
        self.dirpath = dirpath
        self.filename = filename

    def on_train_end(
        self, trainer, pl_module: AnomalyModule
    ) -> None:  # pylint: disable=W0613
        """Call when the train ends.

        Converts the model to ``onnx`` format and then calls OpenVINO's model optimizer to get the
        ``.xml`` and ``.bin`` IR files.
        """
        logger.info("Exporting the model to OpenVINO")
        os.makedirs(self.dirpath, exist_ok=True)
        onnx_path = os.path.join(self.dirpath, self.filename + ".onnx")
        export_convert(
            model=pl_module,
            input_size=self.input_size,
            onnx_path=onnx_path,
            export_path=self.dirpath,
        )
