import time
import sys

from pathlib import Path

import cv2

from tqdm import tqdm
from omegaconf import OmegaConf

from hamacho.core.utils.warning import hide_warns
hide_warns()

from hamacho.core.deploy import TorchInferencer, MultiCategoryTorchInferencer
from hamacho.core.config import get_configurable_parameters
from hamacho.core.utils.callbacks import (
    get_callbacks,
    MetricsConfigurationCallback,
    VisualizerCallback,
    CSVMetricsLoggerCallback,
)
from hamacho.core.post_processing import parse_single_result


def get_single_inferencer(config, model_path):

    callbacks = get_callbacks(config=config)

    metrics_callback = None
    visualizer_callback = None
    csv_callback = None
    for callback in callbacks:
        if isinstance(callback, MetricsConfigurationCallback):
            metrics_callback = callback
        if isinstance(callback, VisualizerCallback):
            visualizer_callback = callback
        if isinstance(callback, CSVMetricsLoggerCallback):
            csv_callback = callback

    inferencer = TorchInferencer(
        config=config,
        model_source=model_path,
        metrics_callback=metrics_callback,
        visualizer_callback=visualizer_callback,
        csv_callback=csv_callback,
    )
    inferencer.warmup()

    return inferencer


def do_single_inference(image_paths, inferencer):

    for image_path in tqdm(image_paths, desc="Predicting"):
        pred = inferencer.predict(image_path)
        print(parse_single_result(pred=pred, return_normalized=True))


def get_multi_inferencer(configs):

    inferencer = MultiCategoryTorchInferencer(
        configs=configs,
        # use the following argument to save all the category
        # outputs in a custom directory
        # inference_save_path="./results/inferencer_results/",
    )
    inferencer.warmup()

    return inferencer


def do_multi_inference(images, inferencer):

    for image, category in tqdm(images, desc="Predicting"):
        t = time.perf_counter()
        pred = inferencer.predict(image, category)
        print(
            "Prediction Time",
            round(time.perf_counter() - t, 4),
            category,
            image if isinstance(image, str) else ""
        )
        print(parse_single_result(pred=pred, return_normalized=True))


if __name__ == "__main__":
    config_path = './results/no-bad-data/patchcore/config.yaml'
    # output results saved during inference can be controlled in the config file
    # from config.project.save_outputs.<task_type>.<output_type>
    save_path = './results/inferencer_results'
    config = get_configurable_parameters(
        config_path=config_path
    )
    Path(save_path).mkdir(exist_ok=True, parents=True)
    config.project.save_root = save_path
    config.project.inference_dir_name = config.dataset.category
    inferencer_type = sys.argv[1]

    if inferencer_type == "single":
        single_inferencer = get_single_inferencer(
            config,
            './results/no-bad-data/patchcore/weights/trained_data.hmc'
        )

        image_paths = [
            './data/canot_Read/bad/001.PNG',
            './data/canot_Read/bad/002.PNG',
            './data/canot_Read/bad/003.PNG',
        ]
        do_single_inference(image_paths, single_inferencer)

    if inferencer_type == "multi":

        conf_paths = (
            "./results/bottle-sealing-surface-jp/patchcore/config.yaml",
            "./results/mvtec/patchcore/toothbrush/config.yaml",
        )
        confs = []
        for conf_path in conf_paths:
            conf = OmegaConf.load(conf_path)
            confs.append(conf)

        multi_inferencer = get_multi_inferencer(confs)

        image_paths = [
            # category image path, category name (from config.dataset.category)
            ('./data/toothbrush/test/defective/000.png', "toothbrush"),
            ('./data/toothbrush/test/defective/001.png', "toothbrush"),
            ('./data/canot_Read/bad/001.PNG', "bottle-sealing-surface-jp"),
            ('./data/toothbrush/test/defective/002.png', "toothbrush"),
            ('./data/toothbrush/test/defective/003.png', "toothbrush"),
            ('./data/toothbrush/test/defective/004.png', "toothbrush"),
            ('./data/canot_Read/bad/003.PNG', "bottle-sealing-surface-jp"),
            ('./data/toothbrush/test/defective/005.png', "toothbrush"),
            ('./data/toothbrush/test/defective/006.png', "toothbrush"),
            ('./data/toothbrush/test/defective/008.png', "toothbrush"),
            ('./data/toothbrush/test/defective/007.png', "toothbrush"),
        ]

        do_multi_inference(image_paths, multi_inferencer)

        # add new category with config file
        multi_inferencer.add_category_config(
            config=OmegaConf.load(
                "./results/mvtec/patchcore/metal_nut/config.yaml"
            )
        )

        # images can be given instead of image paths
        # in this case the images won't be saved using visualizer callback
        # so it will be faster
        images = [
            # category image path, category name (from config.dataset.category)
            (cv2.imread('./data/metal_nut/test/bent/000.png'), "metal_nut"),
            (cv2.imread('./data/toothbrush/test/defective/009.png'), "toothbrush"),
            (cv2.imread('./data/metal_nut/test/bent/003.png'), "metal_nut"),
            (cv2.imread('./data/canot_Read/bad/002.PNG'), "bottle-sealing-surface-jp"),
            (cv2.imread('./data/metal_nut/test/bent/001.png'), "metal_nut"),
        ]

        do_multi_inference(images, multi_inferencer)

        multi_inferencer.add_category_config(
            config=OmegaConf.load(
                './results/no-bad-data/patchcore/config.yaml'
            )
        )
        image_paths = [
            ('./data/canot_Read/bad/001.PNG', "no-bad-data"),
            ('./data/canot_Read/bad/002.PNG', "no-bad-data"),
            ('./data/canot_Read/bad/003.PNG', "no-bad-data"),
        ]
        do_multi_inference(image_paths, multi_inferencer)
