# Copyright 2021-2024 James Diprose & Tuan Chien
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import os.path
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import List, Tuple

import SimpleITK as sitk
import click
import numpy as np
import pandas as pd
import torch
from monai.transforms import LoadImage, SpatialCrop, SaveImage
from torch.utils.data import DataLoader

from deep_mt.config import Config
from deep_mt.csf import calc_csf
from deep_mt.data_cleaning import merge, find_duplicates, check_readable, csv_file_summary, find_missing_dicoms
from deep_mt.dataset import DeepMTDataset
from deep_mt.evaluate import evaluate
from deep_mt.ml_utils import make_transforms
from deep_mt.predict import predict
from deep_mt.train import train_pytorch, train_sklearn, feature_selection
from deep_mt.utils import match_files
from deep_mt.visualise import visualise, visualise_salience


@click.group()
def cli():
    """The deep_mt command line tool"""

    # Setup logging
    logging.basicConfig(level=logging.INFO)


@cli.command("csv-file-summary")
@click.argument("csv-file", type=click.File("r"))
def csv_file_summary_cmd(csv_file):
    """Print a summary of the CSV file"""

    csv_file_summary(csv_file.name)


@cli.command("check-scans-readable")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def check_readable_cmd(path: str):
    """Check that all DICOM scans in a path are readable"""

    check_readable(path)


@cli.command("merge-scans")
@click.argument("src", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("dst", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def merge_cmd(src: str, dst: str):
    """Merge a source and destination dataset when new data is collected"""

    merge(src, dst)


@cli.command("find-duplicate-scans")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def find_duplicates_cmd(path: str):
    """Find duplicate scans"""

    find_duplicates(path)


@cli.command("find-missing-dicoms")
@click.argument("csv-file", type=click.File("r"))
@click.argument("dicoms-path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output-path", type=click.Path(exists=False, file_okay=True, dir_okay=False))
def find_missing_dicoms_cmd(csv_file, dicoms_path: str, output_path: str):
    """Find missing dicoms"""

    find_missing_dicoms(csv_file, dicoms_path, output_path)


@cli.command("calc-csf")
@click.argument("csv-file", type=click.File("r"))
@click.argument("nii-path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output-path", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--csf-min", type=click.INT, default=0, help="The minimum Hounsfield units to consider as CSF")
@click.option("--csf-max", type=click.INT, default=15, help="The maximum Hounsfield units to consider as CSF")
def calc_csf_cmd(csv_file, nii_path: str, output_path: str, csf_min, csf_max):
    """Calculate the CSF volume for all cases"""

    calc_csf(csv_file, nii_path, output_path, csf_min, csf_max)


@cli.command("feature-selection")
@click.argument("config-file", type=click.File("r"))
@click.option("--n-splits", type=click.INT, default=5, help="Number of cross-validator folds.")
@click.option("--n-repeats", type=click.INT, default=3, help="Number of times cross-validator needs to be repeated")
def feature_selection_cmd(config_file, n_splits: int, n_repeats: int):
    """The feature selection command"""

    path = os.path.abspath(config_file.name)
    config = Config.load(path)
    feature_selection(config, n_splits, n_repeats)


@cli.command("train-pytorch")
@click.argument("config-file", type=click.File("r"))
def train_pytorch_cmd(config_file):
    """The train pytorch command"""

    path = os.path.abspath(config_file.name)
    config = Config.load(path)
    train_pytorch(config)


@cli.command("train-sklearn")
@click.argument("config-file", type=click.File("r"))
def train_sklearn_cmd(config_file):
    """The train sklearn command"""

    path = os.path.abspath(config_file.name)
    config = Config.load(path)
    train_sklearn(config)


@cli.command("evaluate-pytorch")
@click.argument("config-file", type=click.File("r"))
@click.option(
    "--subset",
    "subsets",
    type=click.Choice(["train", "valid", "test"]),
    default=["valid", "test"],
    multiple=True,
    help="What subsets to evaluate. Defaults to valid and test.",
)
def evaluate_cmd(config_file, subsets: List[str]):
    """The evaluate command"""

    path = os.path.abspath(config_file.name)
    config = Config.load(path)

    for subset in subsets:
        print(f"Evaluating: {subset}...")
        evaluate(config, subset)

    # Combine evaluation results when multiple subsets specified
    if len(subsets) > 1:
        file_name = "evaluation-results.csv"
        dfs = []
        for subset in subsets:
            df = pd.read_csv(os.path.join(config.experiment_folder, subset, file_name))
            dfs.append(df)

        names = [
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "f1score",
            "precision",
            "recall",
        ]
        for subset, df in zip(subsets, dfs):
            columns = {name: f"{subset}_{name}" for name in names}
            df.rename(columns=columns, inplace=True)

        # Merge dataframe
        df = dfs[0]
        for i in range(1, len(dfs)):
            df = pd.merge(df, dfs[i], on="weights", suffixes=("", "_drop"))
        df.drop(df.filter(regex="_drop$").columns, axis=1, inplace=True)

        # Make a subset of columns
        subset_columns = ["experiment_name", "weights"]
        calc_mean = "valid" in subsets and "test" in subsets
        if calc_mean:
            df["mean"] = df[["valid_accuracy", "test_accuracy", "valid_roc_auc", "test_roc_auc"]].mean(axis=1)
        for subset in subsets:
            subset_columns.append(f"{subset}_accuracy")
            subset_columns.append(f"{subset}_roc_auc")
        if calc_mean:
            subset_columns.append("mean")
        subset_columns.append("weights_file")

        df = df[subset_columns]

        # Save file
        output_path = os.path.join(config.experiment_folder, file_name)
        df.to_csv(
            output_path,
            index=False,
        )
        print(f"Saved results to: {output_path}")


@cli.command("predict-pytorch")
@click.argument("config-file", type=click.File("r"))
@click.argument("weights-file", type=click.File("r"))
def predict_cmd(config_file, weights_file):
    """The evaluate command"""

    path = os.path.abspath(config_file.name)
    config = Config.load(path)
    predict(config, weights_file.name)


@cli.command("visualise")
@click.argument("config-file", type=click.File("r"))
@click.option(
    "--subset",
    "subsets",
    type=click.Choice(["train", "valid", "test"]),
    default=["train", "valid", "test"],
    multiple=True,
    help="What subsets to visualise. Defaults to train, valid and test.",
)
@click.option("--dpi", type=click.INT, default=300, help="The resolution of the figure in dots per inch")
@click.option("--every-n", type=click.INT, default=3, help="Visualise every 'n' slices")
@click.option("--n-cases", type=click.INT, default=None, help="A limit for the number of cases to visualise")
@click.option(
    "--channel",
    type=click.INT,
    default=None,
    help="The channel index to visualise. By default all channels are visualised.",
)
@click.option("--prediction-file", type=click.File("r"), default=None)
def visualise_cmd(config_file, subsets: List[str], dpi: int, every_n: int, n_cases: int, channel: int, prediction_file):
    """The visualise command"""

    path = os.path.abspath(config_file.name)
    config = Config.load(path)
    predictions = None
    if prediction_file is not None:
        predictions = pd.read_csv(prediction_file.name)
    visualise(
        config=config,
        subsets=subsets,
        dpi=dpi,
        every_n=every_n,
        n_cases=n_cases,
        channel=channel,
        predictions=predictions,
    )


@cli.command("visualise-salience")
@click.argument("config-file", type=click.File("r"))
@click.argument("weights-file", type=click.File("r"))
@click.option(
    "--salience-type",
    type=click.Choice(["gradcam", "gradcam++", "occlusion"]),
    default="gradcam",
    help="What type of salience visualisation to use, either gradcam or gradcam++",
)
@click.option(
    "--subset",
    "subsets",
    type=click.Choice(["train", "valid", "test"]),
    default=["train", "valid", "test"],
    multiple=True,
    help="What subsets to visualise salience for. Defaults to train, valid and test.",
)
@click.option(
    "--case-id",
    "case_ids",
    type=click.STRING,
    default=None,
    multiple=True,
    help="Specific case ids to visualise salience for. The subset that these cases belong to must be specified with "
         "the --subset option.",
)
@click.option("--dpi", type=click.INT, default=300, help="The resolution of the figure in dots per inch")
@click.option("--every-n", type=click.INT, default=3, help="Visualise every 'n' slices")
@click.option("--n-cases", type=click.INT, default=None, help="A limit for the number of cases to visualise")
def visualise_salience_cmd(
    config_file, weights_file, salience_type: str, subsets: List[str], case_ids: List[str], dpi: int, every_n: int, n_cases: int
):
    """The visualise salience command"""

    path = os.path.abspath(config_file.name)
    config = Config.load(path)
    visualise_salience(
        config=config,
        weights_file=weights_file.name,
        salience_type=salience_type,
        subsets=subsets,
        case_ids=set(case_ids),
        dpi=dpi,
        every_n=every_n,
        n_cases=n_cases,
    )


@cli.command("scan-stats")
@click.argument("config-file", type=click.File("r"))
@click.option(
    "--subset",
    "subsets",
    type=click.Choice(["train", "valid", "test"]),
    default=["train", "valid", "test"],
    multiple=True,
    help="What subsets to generate scan stats for. Defaults to train, valid and test.",
)
def scan_stats(config_file, subsets: List[str]):
    path = os.path.abspath(config_file.name)
    config = Config.load(path)

    # Load dataset
    dataset = DeepMTDataset(
        csv_path=config.csv_path,
        scan_folder=config.scan_folder,
        target_class=config.target_class,
        ct_key=config.ct_key,
        cta_key=config.cta_key,
        ct_mask_key=config.ct_mask_key,
        stratify_class=config.stratify_class,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
        centre_test_set=config.centre_test_set,
        pre_process_std_scale=config.std_scale,
        pre_process_fill_missing_scans=config.missing_scans.fill,
        pre_process_missing_scans_class=config.missing_scans.class_name,
    )

    sum_data = 0
    sum_squared_data = 0
    count_pixels = 0
    count_images = 0
    min_data = float("inf")
    max_data = float("-inf")

    # Define transforms
    config.transform_name = "deep_mt.transform.make_transforms_baseline"
    config.transform_kwargs["resize"] = False
    transforms, _ = make_transforms(config)

    for subset in subsets:
        # Create datasets
        if subset == "train":
            ds = dataset.pytorch_train(transforms)
        elif subset == "valid":
            ds = dataset.pytorch_valid(transforms)
        else:
            ds = dataset.pytorch_test(transforms)

        # Load data
        loader = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for batch in loader:
            images = batch[config.output_key]
            for image in images:
                image = image.cpu().detach().numpy()
                sum_data += np.sum(image)
                sum_squared_data += (image**2).sum()

                # Calculate number of pixels
                img_pixels = 1
                for value in image.shape:
                    img_pixels *= value
                count_pixels += img_pixels

                image_min = np.min(image)
                image_max = np.max(image)

                if image_min < min_data:
                    min_data = image_min

                if image_max > max_data:
                    max_data = image_max

                count_images += 1
                print(f"Processed {count_images} images")

    # Make stats
    mean = sum_data / count_pixels
    variance = sum_squared_data / count_pixels - mean**2
    standard_deviation = math.sqrt(variance)

    # Print stats
    print(f"Subsets: {subsets}")
    print(f"Mean: {mean}")
    print(f"Std: {standard_deviation}")
    print(f"Min: {min_data}")
    print(f"Max: {max_data}")


def resample(
    input_path: str, output_path: str, new_spacing: Tuple[float, float, float] = (1, 1, 1), interpolator=sitk.sitkLinear
):
    print(f"Resampling {input_path} to {new_spacing[0]}x{new_spacing[1]}x{new_spacing[2]}mm")
    image = sitk.ReadImage(input_path, sitk.sitkFloat32)
    image_spacing = image.GetSpacing()
    image_size = image.GetSize()
    new_size = [
        int(round(image_size_ * image_spacing_ / new_spacing_))
        for image_size_, image_spacing_, new_spacing_ in zip(image_size, image_spacing, new_spacing)
    ]

    new_image = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0,
        image.GetPixelID(),
    )
    # useCompression
    sitk.WriteImage(new_image, output_path, useCompression=True, compressionLevel=9)
    print(f"Saved to: {output_path}")


@cli.command("resample")
@click.argument("path", type=click.Path(file_okay=False, dir_okay=True))
@click.argument("pattern", type=click.STRING)
@click.option("--x", type=click.FLOAT, default=1.0, help="The resolution to resample X, in mm.")
@click.option("--y", type=click.FLOAT, default=1.0, help="The resolution to resample Y, in mm.")
@click.option("--z", type=click.FLOAT, default=1.0, help="The resolution to resample X, in mm.")
def resample_cmd(path, pattern: str, x: float, y: float, z: float):
    # Collate scans to process
    paths = []
    for input_path in match_files(path, pattern):
        base_folder = os.path.dirname(input_path)
        file_name = os.path.basename(input_path).replace(".nii.gz", "")
        output_path = os.path.join(base_folder, f"{file_name}_{x}x{y}x{z}mm.nii.gz")
        paths.append((input_path, output_path))

    # Process scans
    resample_shape = (x, y, z)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for input_path, output_path in paths:
            futures.append(executor.submit(resample, input_path, output_path, new_spacing=resample_shape))
        for future in as_completed(futures):
            future.result()


def crop_scan(file_path: str, roi_start, roi_end):
    print(f"Cropping {file_path} to roi_start={roi_start} roi_end={roi_end}")

    # Load and crop image
    img, meta = LoadImage(ensure_channel_first=True)(file_path)
    img = SpatialCrop(roi_start=roi_start, roi_end=roi_end)(img)

    # Save image
    SaveImage(output_dir=os.path.dirname(file_path), output_postfix="crop", separate_folder=False, resample=False)(img)


@cli.command("crop")
@click.argument("path", type=click.Path(file_okay=False, dir_okay=True))
@click.argument("pattern", type=click.STRING)
@click.argument("roi-start", nargs=3, type=click.Tuple([int, int, int]))
@click.argument("roi-end", nargs=3, type=click.Tuple([int, int, int]))
def crop_cmd(path, pattern, roi_start, roi_end):
    file_names = match_files(path, pattern)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for file_path in file_names:
            futures.append(executor.submit(crop_scan, file_path, roi_start, roi_end))
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    cli()
