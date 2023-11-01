"""Callback that saves model weights from the state dict."""


import logging

import torch
from pytorch_lightning import Callback

from hamacho.plug_in.models.components import AnomalyModule

logger = logging.getLogger(__name__)


class SaveModelCallback(Callback):
    """Callback that loads the model weights from the state dict."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def on_validation_end(
        self, _trainer, pl_module: AnomalyModule
    ) -> None:
        """Call when training ends.
        Saves the model weights from the PyTorch module at the ``weights_path`` 
        """
        logger.info("Saving the model at %s", self.weights_path)
        state_dict: dict = pl_module.get_trained_data()

        if pl_module.need_metrics_in_state_dict:
            metrics_data = pl_module.get_metrics_data()
            state_dict.update(**metrics_data)
    
        torch.save(state_dict, self.weights_path)
