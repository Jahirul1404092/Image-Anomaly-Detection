import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.cuda import CUDAAccelerator

from hamacho.core.data import get_datamodule
from hamacho.plug_in.models import get_model
from hamacho.core.utils.callbacks import (
    get_callbacks,
    CSVMetricsLoggerCallback,
    PlotMetrics,
    VisualizerCallback,
    SigmaSixNormalizationCallback,
)


def run_train_test(config, run_test=False):
    """Runs train(and test) and returns Trainer(and test outputs)"""
    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)
    model = get_model(config)
    datamodule = get_datamodule(config)
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
    for index, callback in enumerate(callbacks):
        if isinstance(callback, SigmaSixNormalizationCallback):
            sigma6_callback = callback
            sigma6_callback.reset_dist_on_validation_end = False

    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)

    out = (trainer,)
    if run_test:
        results = trainer.test(model=model, datamodule=datamodule)
        out += (results,)

    return out


def get_pl_accelerator(accelerator: str) -> Accelerator:
    """Returns torch.device type from str type accelerator"""

    if accelerator == "cpu":
        return CPUAccelerator
    elif accelerator in ("gpu", "auto"):
        cuda_available = torch.cuda.is_available()
        return CUDAAccelerator if cuda_available else CPUAccelerator
