"""Anomaly Score Normalization Callback that uses min-max normalization."""


from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from hamacho.core.post_processing.normalization.min_max import normalize
from hamacho.plug_in.models.components import AnomalyModule


class MinMaxNormalizationCallback(Callback):
    """Callback that normalizes the image-level and pixel-level anomaly scores using min-max normalization."""

    def __init__(
        self,
        norm_image_threshold: Optional[float] = None,
        norm_pixel_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.norm_image_threshold = norm_image_threshold
        self.norm_pixel_threshold = norm_pixel_threshold

    def on_test_start(self, _trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the test begins."""
        self._set_thresholds(pl_module)

    def on_predict_start(self, _trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the test begins."""
        self._set_thresholds(pl_module)

    def _set_thresholds(self, pl_module: AnomalyModule) -> None:
        if pl_module.image_metrics is not None \
            and self.norm_image_threshold is not None:
            pl_module.image_metrics.set_threshold(self.norm_image_threshold)
        elif pl_module.image_metrics is not None:
            pl_module.image_metrics.set_threshold(0.5)

        if pl_module.pixel_metrics is not None \
            and self.norm_pixel_threshold is not None:
            pl_module.pixel_metrics.set_threshold(self.norm_pixel_threshold)
        elif pl_module.pixel_metrics is not None:
            pl_module.pixel_metrics.set_threshold(0.5)

    def on_validation_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends, update the min and max observed values."""
        if "anomaly_maps" in outputs.keys():
            pl_module.min_max(outputs["anomaly_maps"])
        else:
            pl_module.min_max(outputs["pred_scores"])

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
        stats = pl_module.min_max.cpu()
        # outputs["anomaly_maps_denormalized"] = outputs["anomaly_maps"].clone()
        outputs["pred_scores"] = normalize(
            outputs["pred_scores"],
            pl_module.image_threshold.value.cpu(),
            stats.min,
            stats.max,
        )
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = normalize(
                outputs["anomaly_maps"],
                pl_module.pixel_threshold.value.cpu(),
                stats.min,
                stats.max,
            )
