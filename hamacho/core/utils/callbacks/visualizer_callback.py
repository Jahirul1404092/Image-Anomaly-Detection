"""Visualizer Callback."""


import os
import torch
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import cast
from typing import Iterable
from typing import Dict
from typing import Set
from warnings import warn

from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from skimage.segmentation import mark_boundaries

from hamacho.core.post_processing import (
    Visualizer,
    add_anomalous_label,
    add_normal_label,
    compute_mask,
    superimpose_anomaly_map,
    anomaly_map_to_grayscale,
)
from hamacho.core.pre_processing.transforms import Denormalize
from hamacho.core.utils import loggers
from hamacho.core.utils.loggers.base import ImageLoggerBase
from hamacho.plug_in.models.components import AnomalyModule


class VisualizerCallback(Callback):
    """Callback that visualizes the inference results of a model.

    The callback generates a figure showing the original image, the ground truth segmentation mask,
    the predicted error heat map, and the predicted segmentation mask.

    To save the images to the filesystem, add the 'local' keyword to the `project.log_images_to` parameter in the
    config.yaml file.
    """

    def __init__(
        self,
        save_root: str,
        task: str,
        test_dir_name: str,
        inference_dir_name: str,
        test_save_outputs: Optional[Iterable[str]] = None,
        pred_save_outputs: Optional[Iterable[str]] = None,
        log_images_to: Optional[List[str]] = None,
        inputs_are_normalized: bool = True,
        add_label_on_image: bool = True,
        save_combined_result_image: bool = True,
        
    ):
        """Visualizer callback."""
        self.task = task
        self.save_root = save_root
        self.log_images_to = [] if log_images_to is None else log_images_to
        self.inputs_are_normalized = inputs_are_normalized
        self.test_dir_name = test_dir_name
        self.inference_dir_name = inference_dir_name
        test_save_outputs = [] if test_save_outputs is None else test_save_outputs
        pred_save_outputs = [] if pred_save_outputs is None else pred_save_outputs
        self.test_save_outputs = set(test_save_outputs)
        self.pred_save_outputs = set(pred_save_outputs)
        self.add_label_on_image = add_label_on_image
        self.save_combined_result_image = save_combined_result_image

    def _save_images(
        self,
        visualizer: Visualizer,
        module: AnomalyModule,
        trainer: pl.Trainer,
        filename: Path,
        stage_save_dir_name: str,
    ):
        """Save image to logger/local storage.

        Saves the image in `visualizer.figure` to the respective loggers and local storage if specified in
        `log_images_to` in `config.yaml` of the models.

        Args:
            visualizer (Visualizer): Visualizer object from which the `figure` is saved/logged.
            module (AnomalyModule): Anomaly module.
            trainer (Trainer): Pytorch Lightning trainer which holds reference to `logger`
            filename (Path): Path of the input image. This name is used as name for the generated image.
        """
        # Store names of logger and the logger in a dict
        available_loggers = {
            type(logger).__name__.lower().rstrip("logger").lstrip("hamacho"): logger
            for logger in trainer.loggers
        }
        # save image to respective logger
        for log_to in self.log_images_to:
            if log_to in loggers.AVAILABLE_LOGGERS:
                # check if logger object is same as the requested object
                if log_to in available_loggers and isinstance(
                    available_loggers[log_to], ImageLoggerBase
                ):
                    logger: ImageLoggerBase = cast(
                        ImageLoggerBase, available_loggers[log_to]
                    )  # placate mypy
                    logger.add_image(
                        image=visualizer.figure,
                        name=filename.parent.name + "_" + filename.name,
                        global_step=module.global_step,
                    )
                else:
                    warn(
                        f"Requested {log_to} logging but logger object is of type: {type(module.logger)}."
                        f" Skipping logging to {log_to}"
                    )
            else:
                warn(f"{log_to} not in the list of supported image loggers.")

        if "local" in self.log_images_to:
            visualizer.save_image(
                path_to_parent_folder=Path(self.save_root)
                                       / stage_save_dir_name
                                       / "images",
                parent_folder_name=filename.parent.name,
                filename=filename.name,
            )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: Optional[STEP_OUTPUT],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        self._add_images(
            trainer, pl_module, outputs, "test", self.test_dir_name, self.test_save_outputs
        )

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: Optional[STEP_OUTPUT],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        self._add_images(
            trainer, pl_module, outputs, "pred", self.inference_dir_name, self.pred_save_outputs
        )

    def _add_images(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: Optional[STEP_OUTPUT],
        stage: str,
        save_dir_name: str,
        save_outputs: Set[str],
    ) -> None:
        """Log images at the end of every batch.

        Args:
            trainer (Trainer): Pytorch lightning trainer object (unused).
            pl_module (LightningModule): Lightning modules derived from BaseAnomalyLightning object as
            currently only they support logging images.
            outputs (Dict[str, Any]): Outputs of the current test step.
            _batch (Any): Input batch of the current test step (unused).
            _batch_idx (int): Index of the current test batch (unused).
            _dataloader_idx (int): Index of the dataloader that yielded the current batch (unused).
        """
        assert outputs is not None

        if self.inputs_are_normalized:
            normalize = False  # anomaly maps are already normalized
        else:
            normalize = True  # raw anomaly maps. Still need to normalize

        save_combined = self.save_combined_result_image
        save_combined = save_combined if save_outputs else False

        if self._do_pred_mask_computation(outputs, save_outputs):
            # initialize pred masks
            outputs["pred_masks"] = torch.zeros_like(
                outputs["anomaly_maps"]
            )

        threshold = np.float32(pl_module.pixel_metrics.threshold)
        for i, (filename, image, anomaly_map, pred_score) in enumerate(
            zip(
                outputs["image_path"],
                outputs["image"],
                outputs["anomaly_maps"],
                outputs["pred_scores"],
            )
        ):
            image_name = os.path.basename(filename)
            image = Denormalize()(image.cpu())
            anomaly_map: np.ndarray = anomaly_map.cpu().numpy()
            gt_label = None
            if stage == "test":
                gt_label = outputs["label"][i]

            visualizer = Visualizer(save_combined)

            if self.task == "segmentation":

                gt_mask = None
                if "mask" in outputs:
                    gt_mask = outputs["mask"][i].cpu().numpy()
                    gt_mask = gt_mask * 255

                self._add_segmentation_results(
                    visualizer=visualizer,
                    save_outputs=save_outputs,
                    image=image,
                    anomaly_map=anomaly_map,
                    outputs=outputs,
                    output_idx=i,
                    pixel_thresh=threshold,
                    normalize=normalize,
                    gt_mask=gt_mask,
                )

            elif self.task == "classification":

                self._add_classification_results(
                    visualizer=visualizer,
                    threshold=threshold,
                    pred_score=pred_score,
                    image=image,
                    anomaly_map=anomaly_map,
                    save_outputs=save_outputs,
                    normalize=normalize,
                    stage=stage,
                    gt_label=gt_label,
                )

            visualizer.generate(image_name)
            self._save_images(visualizer, pl_module, trainer, Path(filename), save_dir_name)
            visualizer.close()

    def on_test_end(self, _trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Sync logs.

        ``WandbLogger`` was used to be called from this method. This was because logging as a single batch
        ensures that all images appear as part of the same step.

        Args:
            _trainer (pl.Trainer): Pytorch Lightning trainer (unused)
            pl_module (AnomalyModule): Anomaly module
        """
        # if pl_module.logger is not None and isinstance(
        #     pl_module.logger, WandbLogger
        # ):
        #     pl_module.logger.save()
        pass

    def _add_segmentation_results(
        self,
        visualizer: Visualizer,
        save_outputs: Set[str],
        image: np.ndarray,
        anomaly_map: np.ndarray,
        outputs: Dict[str, torch.Tensor],
        output_idx: int,
        pixel_thresh: np.float32,
        normalize: bool,
        gt_mask: Optional[np.ndarray] = None,
    ):
        if "input_image" in save_outputs:
            visualizer.add_image(image=image, title="Input Image",
                                 save_individually=True)

        if "ground_truth_mask" in save_outputs and gt_mask is not None:
            visualizer.add_image(
                image=gt_mask, color_map="gray", title="Ground Truth Mask"
            )

        if "predicted_heat_map" in save_outputs:
            heat_map = superimpose_anomaly_map(
                anomaly_map, image, normalize=normalize
            )
            visualizer.add_image(image=heat_map, title="Predicted Heat Map",
                                 save_individually=True)

        if "grayscale" in save_outputs:
            grayscale_map = anomaly_map_to_grayscale(anomaly_map)
            visualizer.add_image(image=grayscale_map, title="Grayscale",
                                 color_map="gray", save_individually=True)

        if self._do_pred_mask_computation(outputs, save_outputs):
            pred_mask = compute_mask(anomaly_map, pixel_thresh)
            outputs["pred_masks"][output_idx] = torch.from_numpy(pred_mask) / 255

        if "predicted_mask" in save_outputs:
            visualizer.add_image(
                image=pred_mask, color_map="gray", title="Predicted Mask",
                save_individually=True
            )

        if "segmentation_result" in save_outputs:
            vis_img = mark_boundaries(image, pred_mask, color=(1, 0, 0), mode="thick")
            visualizer.add_image(image=vis_img, title="Segmentation Result")

        if "histogram" in save_outputs:
            visualizer.add_histogram(anomaly_map, gt_mask=gt_mask,
                                     title="Histogram", thresh=pixel_thresh)

    def _add_classification_results(
        self,
        visualizer: Visualizer,
        threshold: np.float32,
        pred_score: torch.Tensor,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        save_outputs: Set[str],
        normalize: bool,
        stage: str,
        gt_label: Optional[torch.Tensor],
    ):
        if "input_image" in save_outputs:
            gt_im = image
            gt_im_title = "Input Image"
            if stage == "test" and self.add_label_on_image:
                gt_im = (
                    add_anomalous_label(image) if gt_label
                                else add_normal_label(image)
                )
                gt_im_title = "Image GT label"

            visualizer.add_image(gt_im, title=gt_im_title, save_individually=True)

        if "prediction" in save_outputs:
            heat_map = superimpose_anomaly_map(
                anomaly_map, image, normalize=normalize
            )
            if self.add_label_on_image:
                if pred_score >= threshold:
                    heat_map = add_anomalous_label(heat_map, pred_score)
                else:
                    heat_map = add_normal_label(heat_map, 1 - pred_score)

            visualizer.add_image(image=heat_map, title="Prediction",
                                 save_individually=True)

        if "histogram" in save_outputs:
            visualizer.add_histogram(anomaly_map, gt_mask=None,
                                     title="Histogram", thresh=threshold)

    def _do_pred_mask_computation(
        self,
        outputs: Dict[str, torch.Tensor],
        save_outputs: Set[str],
    ):
        return (
            "mask" in outputs
            or "predicted_mask" in save_outputs
            or "segmentation_result" in save_outputs
        )
