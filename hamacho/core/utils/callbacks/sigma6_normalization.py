"""Anomaly Score Normalization Callback that uses Sigma6 normalization."""


import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from hamacho.core.post_processing.normalization.min_max import normalize
from hamacho.plug_in.models.components import AnomalyModule


logger = logging.getLogger(__name__)


class SigmaSixNormalizationCallback(Callback):
    """Callback that normalizes the image-level and pixel-level anomaly scores using Sigma6 min-max normalization."""

    def __init__(
        self,
        norm_image_threshold: Optional[float] = None,
        norm_pixel_threshold: Optional[float] = None,
        reset_dist_on_validation_end: bool = True,
    ) -> None:
        super().__init__()
        self.norm_image_threshold = norm_image_threshold
        self.norm_pixel_threshold = norm_pixel_threshold
        self.max_override_multiplier = 2
        self.reset_dist_on_validation_end = reset_dist_on_validation_end

    def on_test_start(self, _trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the test begins."""
        self._set_thresholds(pl_module)

    def on_predict_start(self, _trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the test begins."""
        self._set_thresholds(pl_module)

    def _set_thresholds(self, pl_module: AnomalyModule) -> None:
        if (
            pl_module.image_metrics is not None
            and self.norm_image_threshold is not None
        ):
            pl_module.image_metrics.set_threshold(self.norm_image_threshold)
        elif pl_module.image_metrics is not None:
            pl_module.image_metrics.set_threshold(0.5)

        if (
            pl_module.pixel_metrics is not None
            and self.norm_pixel_threshold is not None
        ):
            pl_module.pixel_metrics.set_threshold(self.norm_pixel_threshold)
        elif pl_module.pixel_metrics is not None:
            pl_module.pixel_metrics.set_threshold(0.5)

    def _override_adaptive_thresholds(self, pl_module: AnomalyModule) -> None:
        """Overrides optimal F1-based adaptive thresholds to validation
        data distribution based thresholds that follows sigma6 rule.
        """
        dist_stat = pl_module.training_distribution.cpu()
        pl_module.image_threshold.value = dist_stat.image_mean + 3 * dist_stat.image_std
        pl_module.pixel_threshold.value = pl_module.image_threshold.value

    def _override_max_stat(self, pl_module: AnomalyModule) -> None:
        """
        Overrides the max value of MinMax metric
        """
        pl_module.min_max.max = (
            pl_module.image_threshold.value * self.max_override_multiplier
        )

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: AnomalyModule
    ) -> None:
        """Called when the validation starts after training.
        This is needed after every epoch, because the statistics must be
        stored in the state dict of the checkpoint file.
        """
        logger.info(
            "Collecting the statistics of the validation training data to normalize the scores."
        )
        pl_module.training_distribution.reset()

    def on_validation_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends, update the min and max observed values
        and keeps track of anomaly scores."""
        if "anomaly_maps" in outputs.keys():
            pl_module.min_max(outputs["anomaly_maps"])
            pl_module.training_distribution.update(anomaly_maps=outputs["anomaly_maps"])
        else:
            pl_module.min_max(outputs["pred_scores"])

        pl_module.training_distribution.update(anomaly_scores=outputs["pred_scores"])

    def on_validation_epoch_end(
        self, _trainer: pl.Trainer, pl_module: AnomalyModule
    ) -> None:
        """Called when the validation epoch ends.
        Computes the anomaly score distribution statistics of the validation data.
        """
        pl_module.training_distribution.compute(apply_log_transform=False)
        self._override_adaptive_thresholds(pl_module)
        self._override_max_stat(pl_module)
        if self.reset_dist_on_validation_end:
            pl_module.training_distribution.reset()

    def on_test_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends, normalizes the predicted scores and anomaly maps."""
        self._normalize_batch(outputs, pl_module)

    def on_predict_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: Dict,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends, normalizes the predicted scores and anomaly maps."""
        self._normalize_batch(outputs, pl_module)

    @staticmethod
    def _normalize_batch(outputs, pl_module):
        """Normalize a batch of predictions."""
        minmax_stat = pl_module.min_max.cpu()
        image_threshold = pl_module.image_threshold.value.cpu()
        outputs["pred_scores"] = normalize(
            outputs["pred_scores"],
            image_threshold,
            minmax_stat.min,
            minmax_stat.max,
        )
        if "anomaly_maps" in outputs.keys():
            pixel_threshold = pl_module.pixel_threshold.value.cpu()
            outputs["anomaly_maps"] = normalize(
                outputs["anomaly_maps"],
                pixel_threshold,
                minmax_stat.min,
                minmax_stat.max,
            )
