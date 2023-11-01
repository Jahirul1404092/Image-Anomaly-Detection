
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, Any

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics.classification import BinaryAUROC, BinaryConfusionMatrix, BinaryROC

from hamacho.plug_in.models import AnomalyModule


class PlotMetrics(Callback):
    """
    Plot required metrics and save as image.
    """
    def __init__(
        self,
        save_root: str,
        test_dir_name: str,
    ) -> None:
        self.save_root = save_root
        self.test_dir_name = test_dir_name
        self.do_roc = False
        self.do_auroc = False
        self.do_cm = False
        self.has_pxl_roc = False
        self.pred_scores = torch.tensor(tuple(), dtype=torch.float32)
        self.gt_labels = torch.tensor(tuple(), dtype=torch.float32)
        # TODO: add only required plots set in config file of model
        # add a section for this in config file (ex. config.metrics.plot: ["auc",])
        plots = ["roc", "auroc", "confusion_matrix"]
        if "roc" in plots:
            self.do_roc = True
            self.im_roc = BinaryROC(pos_label=1)
            self.pxl_roc = BinaryROC(pos_label=1)
        if "auroc" in plots:
            self.do_auroc = True
            self.im_auroc = BinaryAUROC(pos_label=1)
            self.pxl_auroc = BinaryAUROC(pos_label=1)
        if "confusion_matrix" in plots:
            self.cm = True
            self.conf_mat = BinaryConfusionMatrix()

    def on_test_start(
        self,
        trainer: Trainer,
        _pl_module: AnomalyModule
    ) -> None:
        self.save_path = (
            Path(self.save_root)
            / self.test_dir_name
            / "metrics"
        )

    def on_test_batch_end(
        self,
        _trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: Optional[STEP_OUTPUT],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int
    ) -> None:
        im_thresh = pl_module.image_metrics.threshold
        pred_scores = outputs["pred_scores"]
        gt_labels = outputs["label"]

        pred_labels = (pred_scores >= im_thresh).int()

        self.im_roc(pred_scores, gt_labels)
        self.im_auroc(pred_scores, gt_labels)
        self.conf_mat(pred_labels, gt_labels)
        self.pred_scores = torch.cat((self.pred_scores, pred_scores))
        self.gt_labels = torch.cat((self.gt_labels, gt_labels))

        if "mask" in outputs:
            anomaly_maps = outputs["anomaly_maps"]
            am_size = anomaly_maps.size()
            anomaly_maps = anomaly_maps.reshape((-1, *am_size[-2:]))
            gt_masks = outputs["mask"]
            self.pxl_roc(anomaly_maps, gt_masks.int())
            self.pxl_auroc(anomaly_maps, gt_masks.int())
            self.has_pxl_roc = True

    def on_test_end(
        self,
        _trainer: Trainer,
        pl_module: AnomalyModule
    ) -> None:
        cm = self.conf_mat.compute()
        cm_fig = self._plot_cm(cm=cm)
        self._save_fig(fig=cm_fig, filename="confusion_matrix")

        # plot image roc
        print("INFO: ploting Image level ROC curve")
        im_roc = self.im_roc.compute()
        im_auroc = self.im_auroc.compute()
        fpr, tpr, _ = im_roc
        im_fig = plt.figure()
        self._plot_roc_w_auroc(fpr, tpr, im_auroc,
                               title="Image Level ROC Curve")
        self._save_fig(fig=im_fig, filename="image-level-roc")

        img_thresh = pl_module.image_metrics.threshold
        hist_fig = plt.figure()
        self._plot_scores_hist(title="Scores Histogram",
                               threshold=img_thresh)
        self._save_fig(fig=hist_fig, filename="image-level-scores-histogram")

        # plot pixel roc
        if self.has_pxl_roc:
            print("INFO: ploting Pixel level ROC curve")
            pxl_roc = self.pxl_roc.compute()
            pxl_auroc = self.pxl_auroc.compute()
            fpr, tpr, _ = pxl_roc
            pxl_fig = plt.figure()
            self._plot_roc_w_auroc(fpr, tpr, pxl_auroc,
                                   title="Pixel Level ROC Curve")
            self._save_fig(fig=pxl_fig, filename="pixel-level-roc")

    def _plot_cm(
        self,
        cm: torch.Tensor,
    ):
        cm_disp = ConfusionMatrixDisplay(
            confusion_matrix=cm.cpu().numpy(),
            display_labels=["normal", "abnormal"]
        )
        cm_disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        return cm_disp.figure_

    def _plot_scores_hist(
        self,
        threshold: float,
        title: str,
    ) -> None:
        normal = self.pred_scores[self.gt_labels == 0]
        abnormal = self.pred_scores[self.gt_labels > 0]
        y, _, _ = plt.hist(
            x=(normal, abnormal),
            label=('normal', 'abnormal'),
            bins=len(self.pred_scores),
            range=(0, 1),
            alpha=0.55,
            color=('b', 'r'),
        )
        plt.axvline(
            threshold,
            color='k',
            linestyle="--",
            linewidth=1
        )
        plt.text(
            x=threshold,
            y=y.max(),
            s=f" Threshold: {threshold}",
            ha="left",
            va="center",
        )
        plt.title(title)
        plt.legend()

    def _plot_roc_w_auroc(
        self,
        fpr: torch.Tensor,
        tpr: torch.Tensor,
        auroc: torch.Tensor,
        title: str,
    ) -> None:
        """
        Plot ROC curve with AUROC score
        """
        plt.plot(
            fpr, tpr,
            color="darkorange",
            lw=2,
            label=f"AUROC {auroc:.3f}"
        )
        plt.plot((0, 1), (0, 1), color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")

    def _save_fig(
        self,
        fig: plt.figure,
        filename: str,
    ):
        save_path = self.save_path / f"{filename}.png"
        fig.savefig(save_path)
        plt.close(fig)
