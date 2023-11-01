
import os
import torch

def get_cpu_count() -> int:
    """Gets number of CPU in the system

    Returns:
        int: number of CPU
    """
    return os.cpu_count()


def get_torch_device(accelerator: str) -> torch.device:
    """Returns torch.device type from str type accelerator"""

    if accelerator == "cpu":
        return torch.device(accelerator)
    elif accelerator in ("gpu", "auto"):
        cuda_available = torch.cuda.is_available()
        accelerator = "cuda" if cuda_available else "cpu"
        return torch.device(accelerator)


class DummyTrainer:
    """
    This mimicks the Trainer class of PyTorch Lightning
    and allows us to use the callbacks prepared for the library
    to be used for general purpose as well.
    """
    project_root: str = ''
    loggers = []
