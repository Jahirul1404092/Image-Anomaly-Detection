"""Anomaly Visualization."""


import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional, Tuple
from itertools import zip_longest

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
import numpy as np

from matplotlib.figure import Figure

MATPLOTLIB_SUPPORTED_IMAGE_EXT = plt.gcf().canvas.get_supported_filetypes()
jp_font_dir = (
    Path(__file__).parent.parent.parent
    / "fonts"
    / "japanese"
    / "NotoSansJP-Regular.otf"
)
jp_font_prop = fm.FontProperties(fname=jp_font_dir)


class Visualizer:
    """Anomaly Visualization.

    The visualizer object is responsible for collating all the images passed to it into a single image. This can then
    either be logged by accessing the `figure` attribute or can be saved directly by calling `save()` method.

    Example:
        >>> visualizer = Visualizer()
        >>> visualizer.add_image(image=image, title="Image")
        >>> visualizer.close()
    """

    def __init__(self, save_combined_result_image: bool):

        self.save_combined_result_image = save_combined_result_image
        self.images: List[Dict] = []
        self.has_hist: bool = False
        self.anomaly_map_data: dict

        self.figure: Figure
        self.axis: np.ndarray

    def add_image(
        self,
        image: np.ndarray,
        title: str,
        color_map: Optional[str] = None,
        save_individually: bool = False,
    ):
        """Add image to figure.

        Args:
          image (np.ndarray): Image which should be added to the figure.
          title (str): Image title shown on the plot.
          color_map (Optional[str]): Name of matplotlib color map used to map scalar data to colours. Defaults to None.
        """
        image_data = dict(image=image, title=title, color_map=color_map,
                          save_individually=save_individually)
        self.images.append(image_data)

    def add_histogram(self, anomaly_map: np.ndarray, title: str, thresh: float,
                      gt_mask: np.ndarray = None):
        """Add histogram graph"""
        self.anomaly_map_data = {}
        self.anomaly_map_data["map"] = anomaly_map
        self.anomaly_map_data["title"] = title
        self.anomaly_map_data["thresh"] = thresh
        self.anomaly_map_data["gt_mask"] = gt_mask
        self.has_hist = True

    def generate(self, image_name: str) -> None:
        """Generate the image."""
        if not self.save_combined_result_image:
            return

        if self.has_hist:
            self.generate_with_hist(image_name)
            return

        num_cols = len(self.images)
        figure_size = (num_cols * 3, 3)
        self.figure, self.axis = plt.subplots(1, num_cols, figsize=figure_size)
        self.figure.subplots_adjust(right=0.9)

        axes = self.axis if len(self.images) > 1 else [self.axis]
        for axis, image_dict in zip(axes, self.images):
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)
            axis.imshow(image_dict["image"], image_dict["color_map"], vmin=0, vmax=255)
            axis.title.set_text(image_dict["title"])

    def generate_with_hist(self, image_name: str) -> None:
        """Generate images with histogram on top."""
        num_right_fig = len(self.images)
        multiply_col_by = 2 if num_right_fig > 3 else 3
        multiply_col_by = 4 if num_right_fig == 1 else multiply_col_by

        figsize = ((num_right_fig + 1) * multiply_col_by, 5)
        nrow_rf, ncol_rf = self._get_nrow_ncol(num_right_fig, 3)
        self.figure = plt.figure(constrained_layout=True, figsize=figsize)
        subfigs = self.figure.subfigures(1, 2, wspace=0.05)
        left_fig = subfigs[0].subplots(1, 1)
        right_fig = subfigs[1].subplots(nrow_rf, ncol_rf)

        self._generate_histogram(left_fig)

        if len(self.images) == 1:
            zip_l = zip_longest((right_fig,), self.images)
        else:
            zip_l = zip_longest(right_fig.ravel(), self.images)

        for axis, image_dict in zip_l:
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)
            if image_dict is None:
                axis.set_visible(False)
                continue

            axis.imshow(image_dict["image"], image_dict["color_map"], vmin=0, vmax=255)
            axis.title.set_text(image_dict["title"])

        self.figure.suptitle(image_name, fontproperties=jp_font_prop)

    def show(self):
        """Show image on a matplotlib figure."""
        self.figure.show()

    def save_image(
        self,
        path_to_parent_folder: Path,
        parent_folder_name: str,
        filename: str
    ):
        """Save image.

        Args:
          filename (Path): Filename to save image
        """
        if self.save_combined_result_image:
            save_path = path_to_parent_folder / "combined" / parent_folder_name / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if save_path.suffix.lower().strip(".") not in MATPLOTLIB_SUPPORTED_IMAGE_EXT:
                save_path = os.path.splitext(save_path)[0] + ".png"

            self.figure.savefig(save_path, dpi=100)

        for image_dict in self.images:
            image, title, color_map, do_save = image_dict.values()
            if not do_save:
                continue
            
            title = title.replace(" ", "_").lower()
            save_path = path_to_parent_folder / title / parent_folder_name / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_path.suffix.lower().strip(".") not in MATPLOTLIB_SUPPORTED_IMAGE_EXT:
                save_path = os.path.splitext(save_path)[0] + ".png"

            if title == "grayscale":
                image = image.squeeze(2)

            mpimg.imsave(save_path, image, cmap=color_map)

    def close(self):
        """Close figure."""
        try:
            plt.close(self.figure)
        except AttributeError:
            pass

    def _generate_histogram(
        self,
        fig: Figure,
    ) -> None:
        # ravel the array before indexing the abnormal and normal values
        # otherwise the plot gives wrong output
        anomaly_map: np.ndarray = self.anomaly_map_data["map"].squeeze(0)
        pixel_thresh = self.anomaly_map_data["thresh"]
        gt_mask: np.ndarray = self.anomaly_map_data["gt_mask"]

        if gt_mask is not None:
            abnormal_score = anomaly_map[gt_mask > 0]
            normal_score = anomaly_map[gt_mask == 0]

            lf, *_ = fig.hist(
                (abnormal_score.ravel(), normal_score.ravel()),
                label=["abnormal", "normal"],
                bins=min(128, anomaly_map.shape[-1]),
                range=[0, 1],
                histtype='stepfilled',
                alpha=0.35,
                color=['r', 'b'],
            )
            fig.legend()
        else:
            lf, _, _ = fig.hist(anomaly_map.ravel(),
                                bins=min(128, anomaly_map.shape[-1]),
                                range=[0, 1],
                                alpha=0.7)

        fig.axvline(pixel_thresh, color='k', linestyle="--", linewidth=1)
        fig.text(pixel_thresh,
                      lf.max(),
                      f' Threshold: {pixel_thresh:.3f}',
                      ha='left',
                      va='center')
        fig.set_xlabel("score")
        fig.set_ylabel("freq")
        fig.set_title(self.anomaly_map_data["title"])

    @staticmethod
    def _get_nrow_ncol(
        num_plots: int,
        row_limit: Optional[int] = None,
        col_limit: Optional[int] = None,
    ) -> Tuple[int]:
        if row_limit is not None and col_limit is not None:
            raise ValueError(
                "Both `row_limit` and `col_limit` cannot be set"
            )

        if col_limit is not None:
            raise NotImplementedError(
                "`col_limit` arg functionality has not been implemented yet"
            )

        if num_plots <= row_limit:
            return (num_plots, 1)
        else:
            return (row_limit, np.ceil(num_plots / row_limit).astype(int).item())
