"""Helper to show progress bars with `urlretrieve`, check hash of file."""



import hashlib
import io
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from tqdm import tqdm
from urllib.request import urlretrieve

MVTEC_DATASET_MAIN_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download"

MVTEC_FULL_DATASET_LAST_SECTION = "420938113-1629952094"
MVTEC_FULL_DATASET_TAR_HASH = "eefca59f2cede9c3fc5b6befbfec275e"

MVTEC_CATEGORY_URL_LAST_SECTION = {
    "bottle": "420937370-1629951468",
    "cable": "420937413-1629951498",
    "capsule": "420937454-1629951595",
    "carpet": "420937484-1629951672",
    "grid": "420937487-1629951814",
    "hazelnut": "420937545-1629951845",
    "leather": "420937607-1629951964",
    "metal_nut": "420937637-1629952063",
    "pill": "420938129-1629953099",
    "screw": "420938130-1629953152",
    "tile": "420938133-1629953189",
    "toothbrush": "420938134-1629953256",
    "transistor": "420938166-1629953277",
    "wood": "420938383-1629953354",
    "zipper": "420938385-1629953449",
}

MVTEC_CATEGORY_TAR_HASH = {
    "bottle": "79b4d46b8647d969e323afef8d8445ff",
    "cable": "2ba411739e8239e7558ead13f18b0ead",
    "capsule": "380afc46701c99cb7b9a928edbe16eb5",
    "carpet": "5ce0a646d054d9e88195bd74c690de4b",
    "grid": "37239e47f99673bb73393b67ca4cc418",
    "hazelnut": "5c30a9e574a762d845ca79cf2f241b01",
    "leather": "e7971b60323840bfc36247e8d9c3598e",
    "metal_nut": "bfbc32e780827923f4b2c61923e364f7",
    "pill": "834e2d57f62cd4d8d74cd0a5775cc48b",
    "screw": "2e817e865841f4947db4ded51c13bb23",
    "tile": "d78d2c0f6d8bbe9cbe3b72a77bc3e328",
    "toothbrush": "53ba30bb7aeac103e28e1b83ae28ffb9",
    "transistor": "4fe2681f0ce1793cbf71d762f926d564",
    "wood": "b2478b1c65e9a470f360cc6173299d58",
    "zipper": "015bfd289a1782abc77318aa015a728d",
}


class DownloadProgressBar(tqdm):
    """Create progress bar for urlretrieve. Subclasses `tqdm`.

    For information about the parameters in constructor, refer to `tqdm`'s documentation.

    Args:
        iterable (Optional[Iterable]): Iterable to decorate with a progressbar.
                            Leave blank to manually manage the updates.
        desc (Optional[str]): Prefix for the progressbar.
        total (Optional[Union[int, float]]): The number of expected iterations. If unspecified,
                                            len(iterable) is used if possible. If float("inf") or as a last
                                            resort, only basic progress statistics are displayed
                                            (no ETA, no progressbar).
                                            If `gui` is True and this parameter needs subsequent updating,
                                            specify an initial arbitrary large positive number,
                                            e.g. 9e9.
        leave (Optional[bool]): upon termination of iteration. If `None`, will leave only if `position` is `0`.
        file (Optional[Union[io.TextIOWrapper, io.StringIO]]): Specifies where to output the progress messages
                                                            (default: sys.stderr). Uses `file.write(str)` and
                                                            `file.flush()` methods.  For encoding, see
                                                            `write_bytes`.
        ncols (Optional[int]): The width of the entire output message. If specified,
                            dynamically resizes the progressbar to stay within this bound.
                            If unspecified, attempts to use environment width. The
                            fallback is a meter width of 10 and no limit for the counter and
                            statistics. If 0, will not print any meter (only stats).
        mininterval (Optional[float]): Minimum progress display update interval [default: 0.1] seconds.
        maxinterval (Optional[float]): Maximum progress display update interval [default: 10] seconds.
                                    Automatically adjusts `miniters` to correspond to `mininterval`
                                    after long display update lag. Only works if `dynamic_miniters`
                                    or monitor thread is enabled.
        miniters (Optional[Union[int, float]]): Minimum progress display update interval, in iterations.
                                            If 0 and `dynamic_miniters`, will automatically adjust to equal
                                            `mininterval` (more CPU efficient, good for tight loops).
                                            If > 0, will skip display of specified number of iterations.
                                            Tweak this and `mininterval` to get very efficient loops.
                                            If your progress is erratic with both fast and slow iterations
                                            (network, skipping items, etc) you should set miniters=1.
        use_ascii (Optional[Union[bool, str]]): If unspecified or False, use unicode (smooth blocks) to fill
                                        the meter. The fallback is to use ASCII characters " 123456789#".
        disable (Optional[bool]): Whether to disable the entire progressbar wrapper
                                    [default: False]. If set to None, disable on non-TTY.
        unit (Optional[str]): String that will be used to define the unit of each iteration
                            [default: it].
        unit_scale (Union[bool, int, float]): If 1 or True, the number of iterations will be reduced/scaled
                            automatically and a metric prefix following the
                            International System of Units standard will be added
                            (kilo, mega, etc.) [default: False]. If any other non-zero
                            number, will scale `total` and `n`.
        dynamic_ncols (Optional[bool]): If set, constantly alters `ncols` and `nrows` to the
                                        environment (allowing for window resizes) [default: False].
        smoothing (Optional[float]): Exponential moving average smoothing factor for speed estimates
                                    (ignored in GUI mode). Ranges from 0 (average speed) to 1
                                    (current/instantaneous speed) [default: 0.3].
        bar_format (Optional[str]):  Specify a custom bar string formatting. May impact performance.
                                    [default: '{l_bar}{bar}{r_bar}'], where
                                    l_bar='{desc}: {percentage:3.0f}%|' and
                                    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
                                    '{rate_fmt}{postfix}]'
                                    Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
                                    percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
                                    rate, rate_fmt, rate_noinv, rate_noinv_fmt,
                                    rate_inv, rate_inv_fmt, postfix, unit_divisor,
                                    remaining, remaining_s, eta.
                                    Note that a trailing ": " is automatically removed after {desc}
                                    if the latter is empty.
        initial (Optional[Union[int, float]]): The initial counter value. Useful when restarting a progress
                                            bar [default: 0]. If using float, consider specifying `{n:.3f}`
                                            or similar in `bar_format`, or specifying `unit_scale`.
        position (Optional[int]): Specify the line offset to print this bar (starting from 0)
                                    Automatic if unspecified.
                                    Useful to manage multiple bars at once (eg, from threads).
        postfix (Optional[Dict]): Specify additional stats to display at the end of the bar.
                                    Calls `set_postfix(**postfix)` if possible (dict).
        unit_divisor (Optional[float]): [default: 1000], ignored unless `unit_scale` is True.
        write_bytes (Optional[bool]): If (default: None) and `file` is unspecified,
                                    bytes will be written in Python 2. If `True` will also write
                                    bytes. In all other cases will default to unicode.
        lock_args (Optional[tuple]): Passed to `refresh` for intermediate output
                                    (initialisation, iterating, and updating).
                                    nrows (Optional[int]): The screen height. If specified, hides nested bars
                                    outside this bound. If unspecified, attempts to use environment height.
                                    The fallback is 20.
        colour (Optional[str]): Bar colour (e.g. 'green', '#00ff00').
        delay (Optional[float]): Don't display until [default: 0] seconds have elapsed.
        gui (Optional[bool]): WARNING: internal parameter - do not use.
                                Use tqdm.gui.tqdm(...) instead. If set, will attempt to use
                                matplotlib animations for a graphical output [default: False].


    Example:
        >>> with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as p_bar:
        >>>         urllib.request.urlretrieve(url, filename=output_path, reporthook=p_bar.update_to)
    """

    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        desc: Optional[str] = None,
        total: Optional[Union[int, float]] = None,
        leave: Optional[bool] = True,
        file: Optional[Union[io.TextIOWrapper, io.StringIO]] = None,
        ncols: Optional[int] = None,
        mininterval: Optional[float] = 0.1,
        maxinterval: Optional[float] = 10.0,
        miniters: Optional[Union[int, float]] = None,
        use_ascii: Optional[Union[bool, str]] = None,
        disable: Optional[bool] = False,
        unit: Optional[str] = "it",
        unit_scale: Optional[Union[bool, int, float]] = False,
        dynamic_ncols: Optional[bool] = False,
        smoothing: Optional[float] = 0.3,
        bar_format: Optional[str] = None,
        initial: Optional[Union[int, float]] = 0,
        position: Optional[int] = None,
        postfix: Optional[Dict] = None,
        unit_divisor: Optional[float] = 1000,
        write_bytes: Optional[bool] = None,
        lock_args: Optional[tuple] = None,
        nrows: Optional[int] = None,
        colour: Optional[str] = None,
        delay: Optional[float] = 0,
        gui: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            iterable=iterable,
            desc=desc,
            total=total,
            leave=leave,
            file=file,
            ncols=ncols,
            mininterval=mininterval,
            maxinterval=maxinterval,
            miniters=miniters,
            ascii=use_ascii,
            disable=disable,
            unit=unit,
            unit_scale=unit_scale,
            dynamic_ncols=dynamic_ncols,
            smoothing=smoothing,
            bar_format=bar_format,
            initial=initial,
            position=position,
            postfix=postfix,
            unit_divisor=unit_divisor,
            write_bytes=write_bytes,
            lock_args=lock_args,
            nrows=nrows,
            colour=colour,
            delay=delay,
            gui=gui,
            **kwargs,
        )
        self.total: Optional[Union[int, float]]

    def update_to(
        self, chunk_number: int = 1, max_chunk_size: int = 1, total_size=None
    ):
        """Progress bar hook for tqdm.

        Based on https://stackoverflow.com/a/53877507
        The implementor does not have to bother about passing parameters to this as it gets them from urlretrieve.
        However the context needs a few parameters. Refer to the example.

        Args:
            chunk_number (int, optional): The current chunk being processed. Defaults to 1.
            max_chunk_size (int, optional): Maximum size of each chunk. Defaults to 1.
            total_size ([type], optional): Total download size. Defaults to None.
        """
        if total_size is not None:
            self.total = total_size
        self.update(chunk_number * max_chunk_size - self.n)


def hash_check(file_path: Path, expected_hash: str):
    """Raise assert error if hash does not match the calculated hash of the file.

    Args:
        file_path (Path): Path to file.
        expected_hash (str): Expected hash of the file.
    """
    with open(file_path, "rb") as hash_file:
        assert (
            hashlib.md5(hash_file.read()).hexdigest() == expected_hash
        ), f"Downloaded file {file_path} does not match the required hash."


def get_mvtec_url(category: Union[str, None] = None) -> str:
    if category is None:
        # mvtec_anomaly_detection.tar.xz
        return f"{MVTEC_DATASET_MAIN_URL}/{MVTEC_FULL_DATASET_LAST_SECTION}/mvtec_anomaly_detection.tar.xz"

    elif category in MVTEC_CATEGORY_URL_LAST_SECTION:
        return f"{MVTEC_DATASET_MAIN_URL}/{MVTEC_CATEGORY_URL_LAST_SECTION[category]}/{category}.tar.xz"

    else:
        raise ValueError(f"{category} is not a valid MVTec Dataset category")


def hash_check_category(zip_file_path: str, category: Union[str, None] = None) -> str:
    if category is None:
        hash_check(zip_file_path, MVTEC_FULL_DATASET_TAR_HASH)

    elif category in MVTEC_CATEGORY_TAR_HASH:
        hash_check(zip_file_path, MVTEC_CATEGORY_TAR_HASH[category])

    else:
        raise ValueError(f"{category} is not a valid MVTec Dataset category")

def download_file(url: str, file_path: str, desc: str) -> None:
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=desc
    ) as progress_bar:
        urlretrieve(
            url=url,
            filename=file_path,
            reporthook=progress_bar.update_to,
        )

def extract_tar(file_path: str, extract_dir: str, delete_tar: bool = False):
    with tarfile.open(file_path) as tar_file:
        tar_file.extractall(extract_dir)

    if delete_tar:
        Path(file_path).unlink()
