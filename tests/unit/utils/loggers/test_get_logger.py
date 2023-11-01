"""Tests to ascertain requested logger."""



from unittest.mock import patch

patch("pytorch_lightning.utilities.imports._package_available", False)
patch("pytorch_lightning.loggers.wandb.WandbLogger")

import pytest
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger

from hamacho.core.utils.loggers import (
    TensorBoardLogger,
    UnknownLogger,
    get_experiment_logger,
)


def test_get_experiment_logger():
    """Test whether the right logger is returned."""

    config = OmegaConf.create(
        {
            "project": {"logger": None, "path": "/tmp"},
            "dataset": {"name": "dummy", "category": "cat1"},
            "model": {"name": "DummyModel"},
        }
    )

    with patch("pytorch_lightning.loggers.wandb.wandb"):

        # get no logger
        logger = get_experiment_logger(config=config)
        assert isinstance(logger, bool)
        config.project.logger = False
        logger = get_experiment_logger(config=config)
        assert isinstance(logger, bool)

        # get tensorboard
        config.project.logger = "tensorboard"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], TensorBoardLogger)

        # get csv logger.
        config.project.logger = "csv"
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], CSVLogger)

        # get multiple loggers
        config.project.logger = ["tensorboard", "csv"]
        logger = get_experiment_logger(config=config)
        assert isinstance(logger[0], TensorBoardLogger)
        assert isinstance(logger[1], CSVLogger)

        # raise unknown
        with pytest.raises(UnknownLogger):
            config.project.logger = "randomlogger"
            logger = get_experiment_logger(config=config)
