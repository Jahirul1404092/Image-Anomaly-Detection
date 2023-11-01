"""Common helpers for both nightly and pre-merge model tests."""



import os
from typing import Dict, List, Tuple, Union

import numpy as np
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from hamacho.core.config import get_configurable_parameters, update_nncf_config
from hamacho.core.utils.callbacks import (
    get_callbacks,
    VisualizerCallback,
    PlotMetrics,
    CSVMetricsLoggerCallback,
)
from hamacho.core.data import get_datamodule
from hamacho.plug_in.models import get_model
from hamacho.plug_in.models.components import AnomalyModule


def setup_model_train(
    model_name: str,
    dataset_path: str,
    project_path: str,
    nncf: bool,
    category: str,
    score_type: str = None,
    weight_file: str = "weights/model.ckpt",
    fast_run: bool = False,
    accelerator: str = "auto",
    device: Union[List[int], int] = [0],
) -> Tuple[Union[DictConfig, ListConfig], LightningDataModule, AnomalyModule, Trainer]:
    """Train the model based on the parameters passed.

    Args:
        model_name (str): Name of the model to train.
        dataset_path (str): Location of the dataset.
        project_path (str): Path to temporary project folder.
        nncf (bool): Add nncf callback.
        category (str): Category to train on.
        score_type (str, optional): Only used for DFM. Defaults to None.
        weight_file (str, optional): Path to weight file.
        fast_run (bool, optional): If set to true, the model trains for only 1 epoch. We train for one epoch as
            this ensures that both anomalous and non-anomalous images are present in the validation step.
        device (List[int], int, optional): Select which device you want to train the model on. Defaults to first GPU.

    Returns:
        Tuple[DictConfig, LightningDataModule, AnomalyModule, Trainer]: config, datamodule, trained model, trainer
    """
    config = get_configurable_parameters(model_name=model_name)
    if score_type is not None:
        config.model.score_type = score_type
    config.project.seed = 42
    config.dataset.category = category
    config.dataset.path = dataset_path
    config.project.log_images_to = []
    config.trainer.devices = device
    config.trainer.accelerator = accelerator

    # Remove legacy flags
    for legacy_device in ["num_processes", "gpus", "ipus", "tpu_cores"]:
        if legacy_device in config.trainer:
            config.trainer[legacy_device] = None

    # If weight file is empty, remove the key from config
    if "weight_file" in config.model.keys() and weight_file == "":
        config.model.pop("weight_file")
    else:
        config.model.weight_file = weight_file if not fast_run else "weights/last.ckpt"

    if nncf:
        config.optimization["nncf"] = {
            "apply": True,
            "input_info": {"sample_size": None},
        }
        config = update_nncf_config(config)
        config.init_weights = None

    # reassign project path as config is updated in `update_config_for_nncf`
    config.project.path = project_path

    datamodule = get_datamodule(config)
    model = get_model(config)

    callbacks = get_callbacks(config)

    # Force model checkpoint to create checkpoint after first epoch
    if fast_run == True:
        for index, callback in enumerate(callbacks):
            if isinstance(callback, ModelCheckpoint):
                callbacks.pop(index)
                break
        model_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(config.project.path, "weights"),
            filename="last",
            monitor=None,
            mode="max",
            save_last=True,
            auto_insert_metric_name=False,
        )
        callbacks.append(model_checkpoint)

    # Remove the callbacks that are not needed in this test
    for index, callback in enumerate(callbacks):
        if isinstance(callback, CSVMetricsLoggerCallback):
            callbacks.pop(index)
            break
    for index, callback in enumerate(callbacks):
        if isinstance(callback, PlotMetrics):
            callbacks.pop(index)
            break
    # Train the model.
    if fast_run:
        config.trainer.max_epochs = 1
        config.trainer.check_val_every_n_epoch = 1

    trainer = Trainer(callbacks=callbacks, **config.trainer)
    trainer.fit(model=model, datamodule=datamodule)
    return config, datamodule, model, trainer


def model_load_test(
    config: Union[DictConfig, ListConfig],
    datamodule: LightningDataModule,
    results: Dict,
):
    """Create a new model based on the weights specified in config.

    Args:
        config ([Union[DictConfig, ListConfig]): Model config.
        datamodule (LightningDataModule): Dataloader
        results (Dict): Results from original model.

    """
    loaded_model = get_model(config)  # get new model

    callbacks = get_callbacks(config)

    # Remove the callbacks that are not needed in this test
    for index, callback in enumerate(callbacks):
        if isinstance(callback, VisualizerCallback):
            callbacks.pop(index)
            break
    for index, callback in enumerate(callbacks):
        if isinstance(callback, CSVMetricsLoggerCallback):
            callbacks.pop(index)
            break
    for index, callback in enumerate(callbacks):
        if isinstance(callback, PlotMetrics):
            callbacks.pop(index)
            break

    # create new trainer object with LoadModel callback (assumes it is present)
    trainer = Trainer(callbacks=callbacks, **config.trainer)
    # Assumes the new model has LoadModel callback and the old one had ModelCheckpoint callback
    new_results = trainer.test(model=loaded_model, datamodule=datamodule)[0]
    assert np.isclose(
        results["Image Level AUROC"], new_results["Image Level AUROC"]
    ), ("Loaded model does not yield close performance results. "
        f"{results['Image Level AUROC']} : {new_results['Image Level AUROC']}")
    if config.dataset.task == "segmentation":
        assert np.isclose(
            results["Pixel Level AUROC"], new_results["Pixel Level AUROC"]
        ), "Loaded model does not yield close performance results"
