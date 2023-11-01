"""Test trainer for multiple dataset structure formats"""

import os
import tempfile
import shutil
from pathlib import Path

from click.testing import CliRunner

from hamacho.main import train
from hamacho.core.utils.folder import count_files
from hamacho.core.utils.filelist import get_valid_filelist, get_valid_paired_filelist
from tests.helpers.dataset import TestDataset

NUM_TRAIN = 20
NUM_TEST = 6


class TestDataFormat:
    """Test trainer with different data format arg"""

    test_dir_name = "test_predictions"
    save_image_dir_name = "images"
    images_parent_dir_name = "combined"

    @TestDataset(
        num_train=NUM_TRAIN,
        num_test=NUM_TEST,
        path=None,
        seed=42,
        data_format="folder",
        category_name="shapes",
    )
    def test_folder_format(self, model="patchcore", category="shapes", path=""):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(
                train,
                [
                    "--dataset-root",
                    path,
                    "--image-size",
                    "128",
                    "--category",
                    category,
                    "--result-path",
                    result_path,
                    "--with-mask-label",
                ],
                standalone_mode=False,
            )
            trainer = result.return_value
            assert result.exit_code == 0
            assert trainer.datamodule.mask_dir == "mask"
            assert (
                count_files(
                    os.path.join(
                        result_path,
                        category,
                        model,
                        self.test_dir_name,
                        self.save_image_dir_name,
                        self.images_parent_dir_name,
                        "bad",
                    )
                )
                == NUM_TEST
            )

    @TestDataset(
        num_train=NUM_TRAIN,
        num_test=NUM_TEST,
        path=None,
        seed=42,
        data_format="folder",
        category_name="shapes",
        mask_dir="mmm",
    )
    def test_folder_format_no_mask_data(
        self, model="patchcore", category="shapes", path=""
    ):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(
                train,
                [
                    "--dataset-root",
                    path,
                    "--image-size",
                    "128",
                    "--category",
                    category,
                    "--result-path",
                    result_path,
                    "--with-mask-label",
                ],
                standalone_mode=False,
            )

            assert result.exit_code == 0
            assert "The 'mask' directory is missing" in result.output
            assert "Continue training without mask images?" in result.output
            assert (
                count_files(
                    os.path.join(
                        result_path,
                        category,
                        model,
                        self.test_dir_name,
                        self.save_image_dir_name,
                        self.images_parent_dir_name,
                        "bad",
                    )
                )
                == NUM_TEST
            )

    @TestDataset(
        num_train=NUM_TRAIN,
        num_test=NUM_TEST,
        path=None,
        seed=42,
        data_format="folder",
        category_name="shapes",
    )
    def test_folder_format_missing_mask_img(
        self, model="patchcore", category="shapes", path=""
    ):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as result_path:
            os.remove(os.path.join(path, category, "mask", "000.png"))
            result = runner.invoke(
                train,
                [
                    "--dataset-root",
                    path,
                    "--image-size",
                    "128",
                    "--category",
                    category,
                    "--result-path",
                    result_path,
                    "--with-mask-label",
                ],
                standalone_mode=False,
            )

            assert result.exit_code == 0
            assert (
                "WARNING: no mask image detected for bad sample image"
                " '000.png'" in result.output
            )
            assert (
                count_files(
                    os.path.join(
                        result_path,
                        category,
                        model,
                        self.test_dir_name,
                        self.save_image_dir_name,
                        self.images_parent_dir_name,
                        "bad",
                    )
                )
                == NUM_TEST - 1
            )

    @TestDataset(
        num_train=NUM_TRAIN,
        num_test=NUM_TEST,
        path="./data",
        seed=42,
        data_format="folder",
        category_name="shapes",
    )
    def test_folder_format_no_bad(self, model="padim", category="shapes", path=""):
        bad_data_path = Path(f"{path}/{category}/bad")
        mask_data_path = Path(f"{path}/{category}/mask")
        print(mask_data_path)
        shutil.rmtree(str(bad_data_path))
        shutil.rmtree(str(mask_data_path))
        # print("deleted")
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(
                train,
                [
                    "--dataset-root",
                    path,
                    "--image-size",
                    "128",
                    "--category",
                    category,
                    "--model",
                    model,
                    "--task-type",
                    "classification",
                    "--result-path",
                    result_path,
                    "--no-bad-mode",
                ],
                standalone_mode=False,
            )
            assert result.exit_code == 0
            assert (
                count_files(
                    os.path.join(
                        result_path,
                        category,
                        model,
                        self.test_dir_name,
                        self.save_image_dir_name,
                        self.images_parent_dir_name,
                        "good",
                    )
                )
                == NUM_TRAIN * 0.2
            )

    @TestDataset(
        num_train=NUM_TRAIN,
        num_test=NUM_TEST,
        path="./datasets/",
        seed=42,
        data_format="mvtec",
        category_name="toothbrush",
    )
    def test_mvtec_format(self, model="padim", category="shapes", path="./datasets"):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(
                train,
                [
                    "--dataset-root",
                    path,
                    "--model",
                    model,
                    "--image-size",
                    "128",
                    "--data-format",
                    "mvtec",
                    "--category",
                    category,
                    "--result-path",
                    result_path,
                ],
                standalone_mode=False,
            )
            trainer = result.return_value
            assert result.exit_code == 0
            assert (
                count_files(
                    os.path.join(
                        result_path,
                        "mvtec",
                        model,
                        category,
                        self.test_dir_name,
                        self.save_image_dir_name,
                        self.images_parent_dir_name,
                        "hexagon",
                    )
                )
                == NUM_TEST
            )
            assert (
                count_files(
                    os.path.join(
                        result_path,
                        "mvtec",
                        model,
                        category,
                        self.test_dir_name,
                        self.save_image_dir_name,
                        self.images_parent_dir_name,
                        "star",
                    )
                )
                == NUM_TEST
            )

    @TestDataset(
        num_train=NUM_TRAIN,
        num_test=NUM_TEST,
        path="./datasets/",
        seed=42,
        data_format="filelist",
        category_name="bottle-sealing-surface",
    )
    def test_filelist_format(self, model="patchcore", category="shape", path="./data"):
        runner = CliRunner()
        good_files_list = [
            str(x) for x in (Path(path) / category / "good").glob("*.png")
        ]
        bad_files_list = [str(x) for x in (Path(path) / category / "bad").glob("*.png")]
        mask_files_list = [
            str(x) for x in (Path(path) / category / "mask").glob("*.png")
        ]

        good_files = ",".join(good_files_list)
        bad_files = ",".join(bad_files_list)
        mask_files = ",".join(mask_files_list)

        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(
                train,
                [
                    "--model",
                    model,
                    "--image-size",
                    "128",
                    "--data-format",
                    "filelist",
                    "--category",
                    category,
                    "--result-path",
                    result_path,
                    "--good-file-list",
                    good_files,
                    "--bad-file-list",
                    bad_files,
                    "--mask-file-list",
                    mask_files,
                    "--with-mask-label",
                ],
                standalone_mode=False,
            )
            trainer = result.return_value
            assert result.exit_code == 0
            assert sorted(trainer.datamodule.l_normal) == sorted(good_files_list)
            assert sorted(trainer.datamodule.l_abnormal) == sorted(bad_files_list)
            assert sorted(trainer.datamodule.l_abnormal_mask) == sorted(mask_files_list)
            assert (
                count_files(
                    os.path.join(
                        result_path,
                        category,
                        model,
                        self.test_dir_name,
                        self.save_image_dir_name,
                        self.images_parent_dir_name,
                        "bad",
                    )
                )
                == NUM_TEST
            )

    @TestDataset(
        num_train=NUM_TRAIN,
        num_test=NUM_TEST,
        path=None,
        seed=42,
        data_format="filelist",
        category_name="shapes",
    )
    def test_filelist_format_missing_mask_img(
        self, model="patchcore", category="shapes", path=""
    ):
        runner = CliRunner()
        os.remove(os.path.join(path, category, "mask", "000.png"))

        good_files_list = [
            str(x) for x in (Path(path) / category / "good").glob("*.png")
        ]
        bad_files_list = [str(x) for x in (Path(path) / category / "bad").glob("*.png")]
        mask_files_list = [
            str(x) for x in (Path(path) / category / "mask").glob("*.png")
        ]

        l_normal, l_abnormal, l_abnormal_mask, _ = get_valid_filelist(
            good_files_list, bad_files_list, mask_files_list
        )
        l_abnormal, l_abnormal_mask, _ = get_valid_paired_filelist(
            l_abnormal, l_abnormal_mask
        )
        good_files = ",".join(l_normal)
        bad_files = ",".join(l_abnormal)
        mask_files = ",".join(l_abnormal_mask)

        with tempfile.TemporaryDirectory() as result_path:
            result = runner.invoke(
                train,
                [
                    "--dataset-root",
                    path,
                    "--image-size",
                    "128",
                    "--data-format",
                    "filelist",
                    "--category",
                    category,
                    "--result-path",
                    result_path,
                    "--good-file-list",
                    good_files,
                    "--bad-file-list",
                    bad_files,
                    "--mask-file-list",
                    mask_files,
                    "--with-mask-label",
                ],
                standalone_mode=False,
            )
            trainer = result.return_value
            assert result.exit_code == 0
            assert sorted(trainer.datamodule.l_normal) == sorted(l_normal)
            assert sorted(trainer.datamodule.l_abnormal) == sorted(l_abnormal)
            assert sorted(trainer.datamodule.l_abnormal_mask) == sorted(l_abnormal_mask)
            assert (
                count_files(
                    os.path.join(
                        result_path,
                        category,
                        model,
                        self.test_dir_name,
                        self.save_image_dir_name,
                        self.images_parent_dir_name,
                        "bad",
                    )
                )
                == NUM_TEST - 1
            )
