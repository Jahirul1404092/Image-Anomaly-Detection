import os
import shutil
from contextlib import ContextDecorator
from functools import wraps
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, List, Optional, Union

import numpy as np
from skimage.io import imsave

from tests.helpers.dirs import create_dirs

from .shapes import random_shapes


def get_dataset_path(root: str = "./datasets", dataset: str = "MVTec") -> str:
    """Selects path based on tests in local system or docker image.

    Local install assumes datasets are located in anomaly/datasets/.
    In either case, if the location is empty, the dataset is downloaded again.

    Example:
    Assume that `datasets directory exists in ~/hamacho/,

    >>> get_dataset_path(dataset="MVTec")
    './datasets/MVTec'

    """
    # For docker deployment or a CI that runs on server, dataset directory
    # may not necessarily be located in the repo. Therefore, check 
    # dataset path environment variable.
    if "DATASET_PATH" in os.environ:
        path = os.path.join(os.environ["DATASET_PATH"], dataset)
        return path

    # Initially check if `datasets` directory exists locally and look
    # for the `dataset`. This is useful for local testing.
    path = os.path.join(root, dataset)
    if not os.path.isdir(path):
        create_dirs(path)

    return path



def generate_random_anomaly_image(
    image_width: int,
    image_height: int,
    shapes: List[str] = ["triangle", "rectangle"],
    max_shapes: Optional[int] = 5,
    generate_mask: Optional[bool] = False,
) -> Dict:
    """Generate a random image with the corresponding shape entities.

    Args:
        image_width (int): Width of the image
        image_height (int): Height of the image
        shapes (List[str]): List of shapes to draw in the image. Make sure these are different from `anomalous_shapes`
        max_shapes (int): Maximum shapes of a kind in the image. Defaults to 5.
        max_size (Optional[int], optional): Maximum size of the test shapes. Defaults to 10.
        generate_mask (bool): Toggle between train/test split. Train images are restricted to first quadrant.
                    Also generates the mask for the image. Defaults to False.
    Returns:
        Tuple: image if `train` is False. Else return image, mask
    """

    image: np.ndarray = np.full((image_height, image_width, 3), 255, dtype=np.uint8)

    input_region = [0, 0, image_width - 1, image_height - 1]

    for shape in shapes:
        shape_image = random_shapes(
            input_region,
            (image_height, image_width),
            max_shapes=max_shapes,
            shape=shape,
        )
        image = np.minimum(image, shape_image)  # since white is 255

    result = {"image": image}

    if generate_mask:
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        # if color exists in any channel turn the mask bit white.
        # not sure if there is a better way to do this.
        mask[image[..., 0] < 255] = 255
        mask[image[..., 1] < 255] = 255
        mask[image[..., 2] < 255] = 255
        result["mask"] = mask

    return result


class TestDataset:
    def __init__(
        self,
        num_train: int = 1000,
        num_test: int = 100,
        img_height: int = 128,
        img_width: int = 128,
        max_size: int = 10,
        train_shapes: List[str] = ["triangle", "rectangle"],
        test_shapes: List[str] = ["hexagon", "star"],
        path: Union[str, Path] = None,
        use_mvtec: bool = False,
        seed: int = 0,
        data_format: str = "mvtec",
        category_name: str = "shapes",
        mask_dir: str = "mask",
        remove_dir_on_exit: bool = True,
    ) -> None:
        """Creates a context for Generating Dummy Dataset. Useful for wrapping test functions.
        NOTE: for MVTec AD dataset it does not return a category.
        It is adviced to use a default parameter in the function

        Args:
            num_train (int, optional): Number of training images to generate. Defaults to 1000.
            num_test (int, optional): Number of testing images to generate per category. Defaults to 100.
            img_height (int, optional): Height of the image. Defaults to 128.
            img_width (int, optional): Width of the image. Defaults to 128.
            max_size (Optional[int], optional): Maximum size of the test shapes. Defaults to 10.
            train_shapes (List[str], optional): List of good shapes. Defaults to ["circle", "rectangle"].
            test_shapes (List[str], optional): List of anomalous shapes. Defaults to ["triangle", "ellipse"].
            path (Union[str, Path], optional): Path to MVTec AD dataset. Defaults to "./datasets/MVTec".
            use_mvtec (bool, optional): Use MVTec AD dataset or dummy dataset. Defaults to False.
            seed (int, optional): Fixes seed if any number greater than 0 is provided. 0 means no seed. Defaults to 0.

        Example:
            >>> @TestDataset
            >>> def test_some_function(path, category="leather"):
            >>>     do something
        """
        self.num_train = num_train
        self.num_test = num_test
        self.img_height = img_height
        self.img_width = img_width
        self.max_size = max_size
        self.train_shapes = train_shapes
        self.test_shapes = test_shapes
        self.path = path
        self.use_mvtec = use_mvtec
        self.seed = seed
        self.data_format = data_format
        self.category_name = category_name
        self.mask_dir = mask_dir
        self.remove_dir_on_exit = remove_dir_on_exit

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):

            if "data_format" in kwds:
                data_format = kwds["data_format"]
            else:
                data_format = self.data_format

            # If true, will use MVTech AD dataset for testing.
            # Useful for nightly builds
            if self.use_mvtec:
                return func(*args, path=self.path, **kwds)
            elif self.data_format == "mvtec":
                with GeneratedDummyDataset(
                    num_train=self.num_train,
                    num_test=self.num_test,
                    img_height=self.img_height,
                    img_width=self.img_width,
                    train_shapes=self.train_shapes,
                    test_shapes=self.test_shapes,
                    max_size=self.max_size,
                    seed=self.seed,
                    data_format=data_format,
                    category_name=self.category_name,
                    rmdir_on_exit=self.remove_dir_on_exit,
                    root_dir=self.path,
                ) as dataset_path:
                    kwds["category"] = self.category_name
                    return func(*args, path=dataset_path, **kwds)
            elif self.data_format == "folder":
                with GeneratedDummyDataset(
                    num_train=self.num_train,
                    num_test=self.num_test,
                    img_height=self.img_height,
                    img_width=self.img_width,
                    train_shapes=self.train_shapes,
                    test_shapes=self.test_shapes,
                    max_size=self.max_size,
                    seed=self.seed,
                    data_format=data_format,
                    category_name=self.category_name,
                    root_dir=self.path,
                    mask_dir=self.mask_dir,
                    rmdir_on_exit=self.remove_dir_on_exit
                ) as dataset_path:
                    kwds["category"] = self.category_name
                    return func(*args, path=dataset_path, **kwds)
            elif self.data_format == "filelist":
                with GeneratedDummyDataset(
                    num_train=self.num_train,
                    num_test=self.num_test,
                    img_height=self.img_height,
                    img_width=self.img_width,
                    train_shapes=self.train_shapes,
                    test_shapes=self.test_shapes,
                    max_size=self.max_size,
                    seed=self.seed,
                    data_format=data_format,
                    category_name=self.category_name,
                    root_dir=self.path,
                    mask_dir=self.mask_dir,
                    rmdir_on_exit=self.remove_dir_on_exit
                ) as dataset_path:
                    kwds["category"] = self.category_name
                    return func(*args, path=dataset_path, **kwds)

        return inner


class GeneratedDummyDataset(ContextDecorator):
    """Context for generating dummy shapes dataset.
    Example:
        >>> with GeneratedDummyDataset(num_train=1000,num_test=100) as dataset_path:
        >>>     some_function()

        Args:
            num_train (int, optional): Number of training images to generate. Defaults to 1000.
            num_test (int, optional): Number of testing images to generate per category. Defaults to 100.
            img_height (int, optional): Height of the image. Defaults to 128.
            img_width (int, optional): Width of the image. Defaults to 128.
            max_size (Optional[int], optional): Maximum size of the test shapes. Defaults to 10.
            train_shapes (List[str], optional): List of good shapes. Defaults to ["circle", "rectangle"].
            test_shapes (List[str], optional): List of anomalous shapes. Defaults to ["triangle", "ellipse"].
            seed (int, optional): Fixes seed if any number greater than 0 is provided. 0 means no seed. Defaults to 0.
    """

    def __init__(
        self,
        num_train: int = 1000,
        num_test: int = 100,
        img_height: int = 128,
        img_width: int = 128,
        max_size: Optional[int] = 10,
        train_shapes: List[str] = ["triangle", "rectangle"],
        test_shapes: List[str] = ["star", "hexagon"],
        seed: int = 0,
        root_dir: Union[str, Path] = None,
        data_format: str = "mvtec",
        category_name: str = "shapes",
        mask_dir: str = "mask",
        rmdir_on_exit: bool = True,
    ) -> None:
        self.root_dir = mkdtemp() if root_dir is None else root_dir
        self.num_train = num_train
        self.num_test = num_test
        self.train_shapes = train_shapes
        self.test_shapes = test_shapes
        self.image_height = img_height
        self.image_width = img_width
        self.max_size = max_size
        self.seed = seed
        self.data_format = data_format
        self.category = category_name
        self.mask_dir = mask_dir
        self.rmdir_on_exit = rmdir_on_exit
        self.delete_cat_only = False if root_dir is None else True

    def _generate_dataset(self):
        """Generates dummy dataset in a temporary directory using the same
        convention as MVTec AD."""
        # create train images
        if self.data_format == "mvtec":
            train_path = os.path.join(self.root_dir, self.category, "train", "good")
        elif self.data_format == "folder":
            train_path = os.path.join(self.root_dir, self.category, "good")
        elif self.data_format == "filelist":
            train_path = os.path.join(self.root_dir, self.category, "good")

        os.makedirs(train_path, exist_ok=True)
        for i in range(self.num_train):
            result = generate_random_anomaly_image(
                image_width=self.image_width,
                image_height=self.image_height,
                shapes=self.train_shapes,
                generate_mask=False,
            )
            image = result["image"]
            imsave(
                os.path.join(train_path, f"{i:03}.png"), 
                image, check_contrast=False
                )

        # create test images
        for test_category in self.test_shapes:
            if self.data_format == "mvtec":
                test_path = os.path.join(self.root_dir, self.category, "test", test_category)
                mask_path = os.path.join(
                    self.root_dir, self.category, "ground_truth", test_category
                )
    
            elif self.data_format == "folder":
                test_path = os.path.join(self.root_dir, self.category, "bad")
                mask_path = os.path.join(
                    self.root_dir, self.category, self.mask_dir
                )
            elif self.data_format == "filelist":
                test_path = os.path.join(self.root_dir, self.category, "bad")
                mask_path = os.path.join(
                    self.root_dir, self.category, "mask"
                )
            
            os.makedirs(mask_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)
            
            # anomaly and masks. The idea is to superimpose anomalous shapes on top of correct ones
            for i in range(self.num_test):
                correct_shapes = generate_random_anomaly_image(
                    image_width=self.image_width,
                    image_height=self.image_height,
                    shapes=self.train_shapes,
                    generate_mask=False,
                )

                result = generate_random_anomaly_image(
                    image_width=self.image_width,
                    image_height=self.image_height,
                    shapes=[test_category],
                    generate_mask=True,
                )
 
                correct_shapes = correct_shapes["image"]
                image, mask = result["image"], result["mask"]
                image = np.minimum(image, correct_shapes)  # since 255 is white
                imsave(
                    os.path.join(test_path, f"{i:03}.png"), 
                    image, check_contrast=False
                )

                if self.data_format == "mvtec":
                    mask_name = f"{i:03}_mask.png"
                elif self.data_format in ("folder", "filelist"):
                    mask_name = f"{i:03}.png"
                imsave(
                    os.path.join(mask_path, mask_name),
                    mask,
                    check_contrast=False,
                )

        # good test
        if self.data_format == "mvtec":
            test_good = os.path.join(self.root_dir, self.category, "test", "good")
            os.makedirs(test_good, exist_ok=True)
            for i in range(self.num_test):
                result = generate_random_anomaly_image(
                    image_width=self.image_width,
                    image_height=self.image_height,
                    shapes=self.train_shapes,
                )
                image = result["image"]
                imsave(
                    os.path.join(test_good, f"{i:03}.png"), 
                    image, 
                    check_contrast=False
                    )

    def __enter__(self):
        """Creates the dataset in temp folder."""
        if self.seed > 0:
            np.random.seed(self.seed)
        self._generate_dataset()
        return self.root_dir

    def __exit__(self, _exc_type, _exc_value, _exc_traceback):
        """Cleanup the directory."""
        if self.rmdir_on_exit and self.delete_cat_only:
            shutil.rmtree(os.path.join(self.root_dir, self.category))
        elif self.rmdir_on_exit:
            shutil.rmtree(self.root_dir)
