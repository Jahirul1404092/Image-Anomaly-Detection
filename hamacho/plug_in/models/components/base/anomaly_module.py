"""Base Anomaly Module for Training Task."""


from abc import ABC
from typing import Any, Iterable
from typing import List, Dict
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from torch import Tensor
from torch import nn

from hamacho.core.utils.metrics import (
    AdaptiveThreshold,
    MetricCollection,
    AnomalyScoreDistribution,
    MinMax,
)


class AnomalyModule(pl.LightningModule, ABC):
    """AnomalyModule to train, validate, predict and test images.

    Acts as a base class for all the Anomaly Modules in the library.
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model: nn.Module
        self.loss: Tensor
        self.callbacks: List[Callback]

        self.adaptive_threshold: bool

        self.image_threshold = AdaptiveThreshold().cpu()
        self.pixel_threshold = AdaptiveThreshold().cpu()

        self.training_distribution = AnomalyScoreDistribution().cpu()
        self.min_max = MinMax().cpu()

        # Create placeholders for image and pixel metrics.
        # If set from the config file, MetricsConfigurationCallback will
        #   create the metric collections upon setup.
        self.image_metrics: MetricCollection
        self.pixel_metrics: MetricCollection

    def forward(self, batch):  # pylint: disable=arguments-differ
        """Forward-pass input tensor to the module.

        Args:
            batch (Tensor): Input Tensor

        Returns:
            Tensor: Output tensor from the model.
        """
        return self.model(batch)

    def validation_step(self, batch, batch_idx) -> dict:  # type: ignore  # pylint: disable=arguments-differ
        """To be implemented in the subclasses."""
        raise NotImplementedError

    def predict_step(
        self, batch: Any, batch_idx: int, _dataloader_idx: Optional[int] = None
    ) -> Any:
        """Step function called during :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`.

        By default, it calls :meth:`~pytorch_lightning.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch (Tensor): Current batch
            batch_idx (int): Index of current batch
            _dataloader_idx (int): Index of the current dataloader

        Return:
            Predicted output
        """
        outputs = self.validation_step(batch, batch_idx)
        self._post_process(outputs)
        outputs["pred_labels"] = (
            outputs["pred_scores"] >= self.image_threshold.value
        ).int()
        return outputs

    def test_step(self, batch, _):  # pylint: disable=arguments-differ
        """Calls validation_step for anomaly map/score calculation.

        Args:
          batch (Tensor): Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        return self.validation_step(batch, _)

    def validation_step_end(self, val_step_outputs):  # pylint: disable=arguments-differ
        """Called at the end of each validation step."""
        self._outputs_to_cpu(val_step_outputs)
        self._post_process(val_step_outputs)
        return val_step_outputs

    def test_step_end(self, test_step_outputs):  # pylint: disable=arguments-differ
        """Called at the end of each test step."""
        self._outputs_to_cpu(test_step_outputs)
        self._post_process(test_step_outputs)
        return test_step_outputs

    def validation_epoch_end(self, outputs):
        """Compute threshold and performance metrics.

        Args:
          outputs: Batch of outputs from the validation step
        """
        if self.adaptive_threshold:
            self._compute_adaptive_threshold(outputs)
        self._collect_outputs(self.image_metrics, self.pixel_metrics, outputs)
        self._log_metrics()

    def test_epoch_end(self, outputs):
        """Compute and save anomaly scores of the test set.

        Args:
            outputs: Batch of outputs from the validation step
        """
        self._collect_outputs(self.image_metrics, self.pixel_metrics, outputs)
        self._log_metrics()

    def get_metrics_data_keys(self) -> Iterable[str]:
        return (
            "min_max.min",
            "min_max.max",
            "image_threshold.value",
            "pixel_threshold.value",
            "training_distribution.image_mean",
            "training_distribution.image_std",
            "training_distribution.pixel_mean",
            "training_distribution.pixel_std",
        )

    def get_metrics_data(self) -> Dict[str, Tensor]:
        return {
            "min_max.min": self.min_max.min,
            "min_max.max": self.min_max.max,
            "image_threshold.value": self.image_threshold.value,
            "pixel_threshold.value": self.pixel_threshold.value,
            "training_distribution.image_mean": self.training_distribution.image_mean,
            "training_distribution.image_std": self.training_distribution.image_std,
            "training_distribution.pixel_mean": self.training_distribution.pixel_mean,
            "training_distribution.pixel_std": self.training_distribution.pixel_std,
        }

    def get_trained_data_keys(self) -> Iterable[str]:
        raise NotImplementedError

    def get_trained_data(self) -> Dict[str, Tensor]:
        """Return the trained memory_bank and other data"""
        raise NotImplementedError

    def load_trained_data(self, state_dict: Dict[str, Tensor]) -> None:
        """Load memory_bank and other necessary trained data"""
        raise NotImplementedError

    def _compute_adaptive_threshold(self, outputs):
        self._collect_outputs(self.image_threshold, self.pixel_threshold, outputs)
        self.image_threshold.compute()
        if "mask" in outputs[0].keys() and "anomaly_maps" in outputs[0].keys():
            self.pixel_threshold.compute()
        else:
            self.pixel_threshold.value = self.image_threshold.value

        self.image_metrics.set_threshold(self.image_threshold.value.item())
        self.pixel_metrics.set_threshold(self.pixel_threshold.value.item())

    def _collect_outputs(self, image_metric, pixel_metric, outputs):
        for output in outputs:
            image_metric.cpu()
            image_metric.update(output["pred_scores"], output["label"].int())
            if "mask" in output.keys() and "anomaly_maps" in output.keys():
                pixel_metric.cpu()
                pixel_metric.update(
                    output["anomaly_maps"].flatten(), output["mask"].flatten().int()
                )

    def _post_process(self, outputs, keep_denormalized=True):
        """Compute labels based on model predictions."""

        if "pred_scores" not in outputs and "anomaly_maps" in outputs:
            outputs["pred_scores"] = (
                outputs["anomaly_maps"]
                .reshape(outputs["anomaly_maps"].shape[0], -1)
                .max(dim=1)
                .values
            )

        if keep_denormalized and "pred_scores" in outputs:
            outputs["pred_scores_denormalized"] = outputs["pred_scores"].clone()

    def _outputs_to_cpu(self, output):
        # for output in outputs:
        for key, value in output.items():
            if isinstance(value, Tensor):
                output[key] = value.cpu()

    def _log_metrics(self):
        """Log computed performance metrics."""
        self.log_dict(self.image_metrics)
        if self.pixel_metrics.update_called:
            self.log_dict(self.pixel_metrics)
