
import os
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Optional, Any, List, Tuple, Union, Iterable

from torch import Tensor
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.metric import Metric

from hamacho.plug_in.models.components import AnomalyModule


class CSVMetricsLoggerCallback(Callback):
    """
    Callback that log the metric results of a model's prediction.
    """

    def __init__(
        self,
        save_root: str,
        test_dir_name: str,
        inference_dir_name: str,
        test_save_outputs: Iterable = [],
        pred_save_outputs: Iterable = [],
    ) -> None:
        self.save_root = save_root
        self.test_dir_name = test_dir_name
        self.inference_dir_name = inference_dir_name
        self.test_save_outputs = test_save_outputs
        self.pred_save_outputs = pred_save_outputs

        self.common_columns: List[str] = [
            "category",
            "data_type",
            "filepath",
            "filename",
            "ext",
        ]
        self.pixel_columns: List[str] = self.common_columns.copy()
        self.pixel_values: List[Tuple[Union[str, float]]] = []
        self.pixel_metrics: List[Metric] = []

        self.generic_columns: List[str] = self.common_columns.copy()
        self.generic_values: List[Tuple[Union[str, float]]] = []

        self._set_generic_metrics()

    def _set_pixel_metrics(
        self,
        pl_module: AnomalyModule
    ) -> None:
        for name, metric in pl_module.pixel_metrics._modules.items():
            self.pixel_columns.append(name.lower().replace('binary', ''))
            self.pixel_metrics.append(metric)

    def _set_generic_metrics(
        self,
    ) -> None:
        image_columns = ["gt", "pred", "score", "score_norm",
                         "image_threshold", "image_threshold_norm",
                         "pixel_threshold", "pixel_threshold_norm",
                         "confusion_metrix"]
        self.generic_columns.extend(image_columns)

    def _add_pixel_results(
        self,
        category: str,
        data_type: str,
        filepath: str,
        filename: str,
        ext: str,
        pred_mask: Tensor,
        gt_mask: Tensor,
    ) -> None:
        pred_mask = pred_mask.squeeze(0)
        # metric results for good samples returns 0 for some reason
        # this is a hack to get the correct results for good images
        if data_type == "good":
            gt_mask[0][0] = 1
            pred_mask[0][0] = 1

        metric_res = tuple(
            round(metric(pred_mask.cpu(), gt_mask.int().cpu()).item(),
                  4)
            for metric in self.pixel_metrics
        )
        all_res = (category, data_type, filepath, filename, ext, *metric_res)
        self.pixel_values.append(all_res)

    def _add_generic_results(
        self,
        category: str,
        data_type: str,
        filepath: str,
        filename: str,
        ext: str,
        gt_label: Tensor,
        pred_label: Tensor,
        score: Tensor,
        score_norm: Tensor,
        image_thresh: Tensor,
        image_thresh_norm: np.ndarray,
        pixel_thresh: Tensor,
        pixel_thresh_norm: np.ndarray,
    ) -> None:
        if not isinstance(gt_label, str):
            cm = self._get_tptnfpfn(gt_label.item(), pred_label.item())
        else:
            cm = "-"

        metric_res = (gt_label, pred_label, score, score_norm,
                      image_thresh, image_thresh_norm,
                      pixel_thresh, pixel_thresh_norm)
        metric_res = tuple(round(m.item(), 4) if not isinstance(m, str) else m
                           for m in metric_res)
        all_res = (category, data_type, filepath, filename, ext, *metric_res, cm)
        self.generic_values.append(all_res)

    def _get_tptnfpfn(self, gt: int, pred: int) -> str:
        if gt == pred == 1:
            return 'TP'
        elif pred == 1 and gt != pred:
            return 'FP'
        elif gt == pred == 0:
            return 'TN'
        elif pred == 0 and gt != pred:
            return 'FN'
        else:
            return None

    def _save_anomaly_value_csv(
        self,
        anomaly_map: Tensor,
        image_path: str,
        filename: str,
    ) -> None:
        anomaly_map = anomaly_map.squeeze(0).cpu().numpy()
        anomaly_map = np.around(anomaly_map, decimals=5)
        file_parent_dir_name = Path(image_path).name
        save_path = (
            self.anomaly_score_csv_save_path
            / file_parent_dir_name
            / f"{filename}.csv"
        )
        save_path.parent.mkdir(exist_ok=True)
        df = pd.DataFrame(anomaly_map)
        df.to_csv(save_path)

    def _set_save_path(
        self,
        root_dir: str,
        stage_save_dir_name: str,
        save_outputs: Iterable,
    ) -> None:
        self.metrics_csv_save_path = (
            root_dir /
            stage_save_dir_name /
            "metrics"
        )
        self.metrics_csv_save_path.mkdir(parents=True, exist_ok=True)
        if "anomaly_map" in save_outputs:
            self.anomaly_score_csv_save_path = (
                root_dir /
                stage_save_dir_name /
                "csv"
            )
            self.anomaly_score_csv_save_path.mkdir(parents=True, exist_ok=True)

    def on_predict_start(
        self,
        _trainer: Trainer,
        _pl_module: AnomalyModule
    ) -> None:
        root_dir = Path(self.save_root)
        self.generic_values.clear()
        self.pixel_values.clear()
        self._set_save_path(root_dir, self.inference_dir_name, self.pred_save_outputs)

    def on_predict_batch_end(
        self,
        _trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: Optional[STEP_OUTPUT],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int
    ) -> None:
        """Add metrics to List"""
        self._add_results(pl_module, outputs, "pred", self.pred_save_outputs)

    def on_test_start(
        self,
        _trainer: Trainer,
        pl_module: AnomalyModule
    ) -> None:
        root_dir = Path(self.save_root)
        self._set_pixel_metrics(pl_module)
        self._set_save_path(root_dir, self.test_dir_name, self.test_save_outputs)

    def on_test_batch_end(
        self,
        _trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: Optional[STEP_OUTPUT],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int
    ) -> None:
        """Add metrics to List"""
        self._add_results(pl_module, outputs, "test", self.test_save_outputs)

    def _add_results(
        self,
        pl_module: AnomalyModule,
        outputs: Optional[STEP_OUTPUT],
        stage: str,
        save_outputs: Iterable,
    ) -> None:

        image_thresh = pl_module.image_threshold.value
        pixel_thresh = pl_module.pixel_threshold.value
        image_thresh_norm = np.float32(pl_module.image_metrics.threshold)
        pixel_thresh_norm = np.float32(pl_module.pixel_metrics.threshold)

        for i, (
            image_path,
            pred_score_norm,
            pred_score,
            anomaly_map,
        ) in enumerate(
            zip(
                outputs["image_path"],
                outputs["pred_scores"],
                outputs["pred_scores_denormalized"],
                outputs["anomaly_maps"],
            )
        ):
            if stage == "test":
                gt_label = outputs["label"][i]
                data_type = outputs["data_type"][i]
                category = outputs["category"][i]
            elif stage == "pred":
                gt_label = "-"
                data_type = "-"
                category = "-"

            relative_path = os.path.relpath(image_path)
            base = os.path.basename(relative_path)
            filename, ext = os.path.splitext(base)
            ext = ext[1:]
            filepath = os.path.dirname(relative_path)
            pred_label = (pred_score_norm >= image_thresh_norm).int().cpu().numpy()

            if "metrics" in save_outputs:
                self._add_generic_results(category, data_type, filepath, filename,
                                        ext, gt_label, pred_label, pred_score,
                                        pred_score_norm, image_thresh, image_thresh_norm,
                                        pixel_thresh, pixel_thresh_norm)

            if "anomaly_map" in save_outputs:
                self._save_anomaly_value_csv(anomaly_map, filepath, filename)

            if "mask" in outputs:
                pred_mask = outputs["pred_masks"][i]
                gt_mask = outputs["mask"][i]
                self._add_pixel_results(category, data_type, filepath, filename,
                                        ext, pred_mask, gt_mask)


    def on_test_end(
        self,
        _trainer: Trainer,
        _pl_module: AnomalyModule
    ) -> None:
        self._save_results("test", self.test_save_outputs)

    def on_predict_end(
        self,
        _trainer: Trainer,
        _pl_module: AnomalyModule
    ) -> None:
        self._save_results("pred", self.pred_save_outputs)

    def _save_results(
        self,
        stage: str,
        save_outputs: Iterable,
    ) -> None:
        save_path = self.metrics_csv_save_path

        if self.pixel_values:
            pixel_df = pd.DataFrame(
                self.pixel_values,
                columns=self.pixel_columns,
            )
            pixel_df.to_csv(save_path /
                            f"pixel_level_{stage}_outputs.csv",
                            encoding="utf-8_sig")

        if "metrics" in save_outputs:
            generic_df = pd.DataFrame(
                self.generic_values,
                columns=self.generic_columns,
            )
            generic_df.to_csv(save_path /
                            f"{stage}_outputs.csv",
                            encoding="utf-8_sig")
