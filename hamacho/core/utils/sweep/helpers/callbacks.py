"""Get callbacks related to sweep."""




from typing import List

from pytorch_lightning import Callback

from hamacho.core.utils.callbacks.timer import TimerCallback


def get_sweep_callbacks() -> List[Callback]:
    """Gets callbacks relevant to sweep.

    Args:
        config (Union[DictConfig, ListConfig]): Model config loaded from hamacho

    Returns:
        List[Callback]: List of callbacks
    """
    callbacks: List[Callback] = [TimerCallback()]

    return callbacks
