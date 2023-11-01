"""Load PyTorch Lightning Loggers."""


import logging
import os
from typing import Iterable
from typing import List
from typing import Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import LightningLoggerBase

from .tensorboard import TensorBoardLogger

__all__ = [
    "TensorBoardLogger",
    "configure_logger",
    "get_experiment_logger",
]
AVAILABLE_LOGGERS = ["tensorboard", "csv"]


class UnknownLogger(Exception):
    """This is raised when the logger option in `config.yaml` file is set incorrectly."""


def configure_logger(level: Union[int, str] = logging.INFO):
    """Get console logger by name.

    Args:
        level (Union[int, str], optional): Logger Level. Defaults to logging.INFO.

    Returns:
        Logger: The expected logger.
    """

    if isinstance(level, str):
        level = logging.getLevelName(level)

    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_string, level=level)

    # Set Pytorch Lightning logs to have a the consistent formatting with hamacho.
    for handler in logging.getLogger("pytorch_lightning").handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)


def get_experiment_logger(
    config: Union[DictConfig, ListConfig]
) -> Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]:
    """Return a logger based on the choice of logger in the config file.

    Args:
        config (DictConfig): config.yaml file for the corresponding hamacho model.

    Raises:
        ValueError: for any logger types apart from false and tensorboard

    Returns:
        Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]: Logger
    """
    if config.project.logger in [None, False]:
        return False

    logger_list: List[LightningLoggerBase] = []
    if isinstance(config.project.logger, str):
        config.project.logger = [config.project.logger]

    for logger in config.project.logger:
        if logger == "tensorboard":
            logger_list.append(
                TensorBoardLogger(
                    name="Tensorboard Logs",
                    save_dir=os.path.join(config.project.path, "logs"),
                )
            )
        elif logger == "csv":
            logger_list.append(
                CSVLogger(save_dir=os.path.join(config.project.path, "logs"))
            )
        else:
            raise UnknownLogger(
                f"Unknown logger type: {config.project.logger}. "
                f"Available loggers are: {AVAILABLE_LOGGERS}.\n"
                f"To enable the logger, set `project.logger` to `true` or use one of available loggers in config.yaml\n"
                f"To disable the logger, set `project.logger` to `false`."
            )

    return logger_list
