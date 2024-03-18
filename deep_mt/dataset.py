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

from __future__ import annotations

import logging
import math
import os
from enum import Enum
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from monai.data import Dataset
from monai.data import PersistentDataset
from monai.transforms import Compose
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from deep_mt.ml_utils import (
    split_largest_remainder,
    random_sample,
)

TrainValidTestSplit = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def column_name_selector(*column_names):
    pattern = r"^(?:" + "|".join(list(column_names)) + ")$"
    return make_column_selector(pattern=pattern)


class mRSType(Enum):
    mrs0_6 = "mrs0_6"  # 0, 1, 2, 3, 5, 6
    mrs01_26 = "mrs01_36"  # 0-1 / 2-6
    mrs02_36 = "mrs02_36"  # 0-2 / 3-6
    mrs03_46 = "mrs03_46"  # 0-3 / 4-6
    mrs05_6 = "mrs05_6"  # 0-5 / 6

    @staticmethod
    def class_name(member: mRSType, code: int):
        return mRSType.class_names(member)[code]

    @staticmethod
    def class_names(member: mRSType):
        return MRS_TYPE_TO_CLASS_NAMES_INDEX[member]


MRS_TYPE_TO_CLASS_NAMES_INDEX = {
    mRSType.mrs0_6: ["mRS 0", "mRS 1", "mRS 2", "mRS 3", "mRS 4", "mRS 5", "mRS 6"],
    mRSType.mrs01_26: ["mRS 0-1", "mRS 2-6"],
    mRSType.mrs02_36: ["mRS 0-2", "mRS 3-6"],
    mRSType.mrs03_46: ["mRS 0-3", "mRS 4-6"],
    mRSType.mrs05_6: ["mRS 0-5", "mRS 6"],
}


def mrs_quantize(mrs: int, mrs_type: mRSType):
    """Quantize mRS values based on the mRSType

    :param mrs: the raw mRS value.
    :param mrs_type: the mRSType.
    :return: the quantized value.
    """

    assert 0 <= mrs <= 6, f"mRS must be between 0 and 6 inclusive: {mrs}"

    if mrs_type == mRSType.mrs01_26:
        return int(mrs >= 2)
    if mrs_type == mRSType.mrs02_36:
        return int(mrs >= 3)
    elif mrs_type == mRSType.mrs03_46:
        return int(mrs >= 4)
    elif mrs_type == mRSType.mrs05_6:
        return int(mrs >= 6)
    elif mrs_type == mRSType.mrs0_6:
        return mrs


def standard_split(
    *,
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    stratify_class: str,
    random_seed: int,
) -> TrainValidTestSplit:
    """Use the train_ratio, valid_ratio and test_ratio properties to randomly split the entire dataset,
    without taking into account hospital centre.

    :param df: the dataframe.
    :param train_ratio: the train ratio.
    :param valid_ratio: the validation ratio.
    :param test_ratio: the test ratio.
    :param stratify_class: the class to stratify on.
    :return: the train valid test split.
    """

    # Create splits using largest remainder method
    num_cases = df.shape[0]
    train_size, valid_size, test_size = split_largest_remainder(num_cases, train_ratio, valid_ratio, test_ratio)

    # Get data for stratified sampling
    X = np.array(df["case_id"].tolist())
    y = np.array(df[stratify_class].tolist())

    # Test
    sample_index, out_index = random_sample(X, y, test_size, random_seed)
    df_test = df.iloc[sample_index]

    # Valid
    df_out = df.iloc[out_index]
    X, y = X[out_index], y[out_index]
    sample_index, out_index = random_sample(X, y, valid_size, random_seed)
    df_valid = df_out.iloc[sample_index]

    # Train
    df_train = df_out.iloc[out_index]

    return df_train, df_valid, df_test


def standard_split2(
    *,
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    stratify_class: str,
    random_seed: int,
) -> TrainValidTestSplit:
    num_cases = df.shape[0]
    n_train = int(math.floor(num_cases * train_ratio))
    n_valid = int(math.floor(num_cases * test_ratio))
    valid_bdry = n_train + n_valid

    from random import Random

    r = Random(random_seed)
    idx = np.arange(num_cases)
    r.shuffle(idx)
    idx = idx.tolist()

    train_idx = []
    valid_idx = []
    test_idx = []

    for i, val in enumerate(idx):
        if val < n_train:
            train_idx.append(i)
        elif val < valid_bdry:
            valid_idx.append(i)
        else:
            test_idx.append(i)

    df_train = df.iloc[train_idx]
    df_valid = df.iloc[valid_idx]
    df_test = df.iloc[test_idx]

    return df_train, df_valid, df_test


def centre_split(
    *, df: pd.DataFrame, centre_test_set: str, valid_ratio: float, stratify_class: str, random_seed: int
) -> TrainValidTestSplit:
    """Split dataset so that the hospital centre, given by centre_test_set is used as the test  based on the

    :param df: the dataframe.
    :param centre_test_set: the centre test set key to choose the test set based on.
    :param valid_ratio: the validation ratio.
    :param stratify_class: the class to stratify on.
    :return: the train valid test split.
    """

    # Split into subsets
    num_cases = df.shape[0]
    df_centre_a = df.loc[df["centre"] != centre_test_set]
    df_centre_b = df.loc[df["centre"] == centre_test_set]

    # Create train and valid splits using largest remainder method
    num_akl_cases = df_centre_a.shape[0]
    test_ratio = df_centre_b.shape[0] / num_cases
    valid_ratio = valid_ratio / (1.0 - test_ratio)
    train_ratio = 1 - valid_ratio
    train_size, valid_size = split_largest_remainder(num_akl_cases, train_ratio, valid_ratio)

    # Split into train, test, split
    X = np.array(df_centre_a["case_id"].tolist())
    y = np.array(df_centre_a[stratify_class].tolist())
    sample_index, out_index = random_sample(X, y, train_size, random_seed)
    df_train = df_centre_a.iloc[sample_index]
    df_valid = df_centre_a.iloc[out_index]
    df_test = df_centre_b

    return df_train, df_valid, df_test


def replace_constant(df: pd.DataFrame, column_name: str, replace_map: Dict):
    """Replace values in a Pandas Dataframe column based on a replacement map.

    :param df: the Pandas Dataframe.
    :param column_name: the column to perform replacement in.
    :param replace_map: a dictionary where the keys map to values that the keys should be replaced by.
    :return: the dataframe.
    """

    return df.apply(
        lambda col: col.apply(lambda val: replace_map[val] if val in replace_map else val)
        if col.name == column_name
        else col
    )


def make_pre_processing_pipeline(*, standard_scale: bool = True, remainder: str = "passthrough"):
    """Make a preprocessing pipeline.

    :param standard_scale: whether to standard scale data.
    :param remainder: what to do with the remaining data.
    :return: the pipeline.
    """

    # Binary features
    pipe_binary = Pipeline(
        steps=[
            ("impute_most_frequent", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("label_encode", OrdinalEncoder()),
        ]
    )

    # Categorical features
    steps = [
        (
            "vessel_map",
            FunctionTransformer(
                func=lambda df: replace_constant(df, "vessel", {"a1": "m2", "vert": "basilar", "p1": "basilar"}),
                feature_names_out="one-to-one",
            ),
        ),
        (
            "ethnicity_map",
            FunctionTransformer(
                func=lambda df: replace_constant(
                    df, "ethnicity", {"latin_american": "other", "middle_eastern": "other", "african": "other"}
                ),
                feature_names_out="one-to-one",
            ),
        ),
        ("impute_most_frequent", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ("one_hot", OneHotEncoder()),
    ]
    pipe_categorical = Pipeline(steps=steps)

    # Ordinal features
    steps = [
        (
            "aspects_replace_p",
            FunctionTransformer(
                func=lambda df: replace_constant(df, "aspects", {"p": np.nan}),
                feature_names_out="one-to-one",
            ),
        ),
        ("impute_median", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("to_integer", FunctionTransformer(func=lambda col: col.astype(int), feature_names_out="one-to-one")),
        # convert to integer as these are ordinal
    ]
    if standard_scale:
        steps.append(("scaler", StandardScaler()))
    pipe_ordinal = Pipeline(steps=steps)

    # Continuous features: imputed with mean
    steps = [
        ("impute_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
    ]
    if standard_scale:
        steps.append(
            ("scaler", StandardScaler()),
        )
    pipe_continuous_mean = Pipeline(steps=steps)

    # Continuous features: imputed with median
    steps = [
        ("impute_median", SimpleImputer(missing_values=np.nan, strategy="median")),
    ]
    if standard_scale:
        steps.append(("scaler", StandardScaler()))
    pipe_continuous_median = Pipeline(steps=steps)

    # Continuous features: imputed with most_frequent
    steps = [
        ("impute_most_frequent", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
    ]
    if standard_scale:
        steps.append(("scaler", StandardScaler()))
    pipe_continuous_mode = Pipeline(steps=steps)

    return ColumnTransformer(
        [
            (
                "binary",
                pipe_binary,
                column_name_selector(
                    "af",
                    "chf",
                    "current_or_ex_smoker",
                    "diabetes_mellitus",
                    "dyslipidaemia",
                    "hypertension",
                    "ischaemic_heart_disease",
                    "sex",
                    "stroke",
                    "territory",
                    "tpa",
                    "imaging_class_id",
                ),
            ),
            ("categorical", pipe_categorical, column_name_selector("ethnicity", "vessel", "side_of_occlusion")),
            (
                "ordinal",
                pipe_ordinal,
                column_name_selector(
                    "aspects",
                    "mrs_baseline",
                    "nihss_baseline",
                ),
            ),
            ("continuous_mean", pipe_continuous_mean, column_name_selector("hb", "sbp", "imaging_score")),
            (
                "continuous_median",
                pipe_continuous_median,
                column_name_selector("age", "cr", "csf_ratio", "csf_volume", "brain_volume"),
            ),
            (
                "continuous_mode",
                pipe_continuous_mode,
                column_name_selector(
                    "bsl",
                    "onset_to_groin",
                ),
            ),
        ],
        remainder=remainder,
    )


def combine(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> pd.DataFrame:
    """Combine multiple dataframes.

    :param df_train: the train Pandas dataframes.
    :param df_valid: the valid Pandas dataframes.
    :param df_test: the test Pandas dataframes.
    :return: a single dataframe.
    """

    df_train["subset"] = "train"
    df_valid["subset"] = "valid"
    df_test["subset"] = "test"
    df = pd.concat([df_train, df_valid, df_test])
    df.reset_index(inplace=True, drop=True)
    return df


def split(df: pd.DataFrame) -> TrainValidTestSplit:
    """Split a dataframe into three subsets.

    :param df: the dataframe.
    :return: the split dataframes.
    """

    # Re-split subsets
    df_train = df.loc[df["subset"] == "train"]
    df_valid = df.loc[df["subset"] == "valid"]
    df_test = df.loc[df["subset"] == "test"]

    return df_train, df_valid, df_test


def load_predictions(
    *,
    prediction_path: str,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> TrainValidTestSplit:
    """Loads score and class_id from a predictions CSV file and merges it with the dataframes."""

    # Load data
    df_pred = pd.read_csv(prediction_path)
    df_pred.rename(columns={"class_id": "imaging_class_id", "score": "imaging_score"}, inplace=True)

    # Combine dataframes
    df = combine(df_train, df_valid, df_test)

    # Assert that length, number of cases and the splits are the same
    same_length = len(df_pred) == len(df)
    same_values = (
        df_pred[["case_id", "subset"]]
        .sort_values(by=["case_id"])
        .equals(df[["case_id", "subset"]].sort_values(by=["case_id"]))
    )
    if not same_length or not same_values:
        msg = f"load_predictions: dataset and predictions do not match. Dataset length {len(df)} vs predictions length {len(df_pred)}. Cases and subsets match: {same_values}. Prediction_path={prediction_path}"
        raise ValueError(msg)

    # Merge dataframe
    df = pd.merge(df, df_pred, on="case_id", suffixes=("", "_drop"))
    df.drop(df.filter(regex="_drop$").columns, axis=1, inplace=True)

    return split(df)


def fill_missing_scans(
    *,
    target_class: str,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> TrainValidTestSplit:
    """Fill missing scans.

    :param target_class: the class to stratify on.
    :param df_train: the train split.
    :param df_valid: the valid split.
    :param df_test: the test split.
    :return: the dataset splits with the missing scans filled in.
    """

    # Join subsets
    df = combine(df_train, df_valid, df_test)

    # Search for nearest cases with images for cases when ct or cta are missing
    # subset, target class, location, age
    df_missing = df[(~df["ct_exists"]) | (~df["cta_exists"])]
    for index, row in df_missing.iterrows():
        ct_exists = row["ct_exists"]
        case_id = row["case_id"]
        subset = row["subset"]
        target = row[target_class]
        vessel = row["vessel"]
        side_of_occlusion = row["side_of_occlusion"]
        age = row["age"]
        image_key = "ct"
        if ct_exists:
            image_key = "cta"

        # Queries
        query_basic = (
            (df["case_id"] != case_id)
            & (df[f"{image_key}_exists"])
            & (df["subset"] == subset)
            & (df["side_of_occlusion"] == side_of_occlusion)
            & (df[target_class] == target)
        )
        query_vessel = query_basic & (df["vessel"] == vessel)

        # Try query vessel first
        matches = df[query_vessel]
        matches.reset_index(drop=True, inplace=True)
        if matches.shape[0] == 0:
            # If query vessel fails use query basic
            matches = df[query_basic]
            matches.reset_index(drop=True, inplace=True)
            if matches.shape[0] == 0:
                # If no matches throw error
                msg = (
                    f"fill_missing_scans: error no match found: case_id!={case_id}, "
                    f"image_key={image_key}, subset={subset}, vessel={vessel}, side_of_occlusion={side_of_occlusion}, "
                    f"stratify_class={target}"
                )
                logging.error(msg)
                raise Exception(msg)

        # Update with match
        closest_index = matches["age"].sub(age).abs().idxmin()
        closest = matches.iloc[closest_index]
        full_image_key = f"{image_key}_image"
        closest_image = closest[full_image_key]
        df.loc[df.case_id == case_id, full_image_key] = closest_image
        df.loc[df.case_id == case_id, f"{image_key}_exists"] = True

    return split(df)


class DeepMTDataset:

    TABULAR_X_DROP = [
        "admit_date",
        "case_id",
        "case_no",
        "centre",
        "ct_exists",
        "cta_exists",
        "ct_image",
        "cta_image",
        "ct_mask",
        "subset",
        "tici",
        "mrs",
        "mrs0_6_label",
        "mrs01_36_label",
        "mrs02_36_label",
        "mrs03_46_label",
        "mrs05_6_label",
        "age_median_label",
    ]

    def __init__(
        self,
        *,
        csv_path: str,
        scan_folder: str,
        target_class: str,
        ct_key: str,
        cta_key: str | None,
        ct_mask_key: str,
        cache_folder: str = None,
        stratify_class: str = None,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_seed: int = None,
        centre_test_set: str = None,
        pre_process_std_scale: bool = False,
        pre_process_fill_missing_scans: bool = False,
        pre_process_missing_scans_class: str = None,
        prediction_path: str = None,
        features: List[str] = None,
    ):
        """Make a stroke dataset.

        :param csv_path: the path to the CSV file containing tabular data.
        :param scan_folder: the path to the scan folder.
        :param target_class: the target class to train the model to predict.
        :param ct_key: the CT scan key.
        :param cta_key: the CTA scan key.
        :param ct_mask_key: the CT mask key.
        :param stratify_class: an optional class to stratify on instead of the target_class. Used when the target
        class is automatically generated based on another class, e.g. when mrs is transformed to mrs02_36, you still
        want to stratify and impute missing images based on the raw mrs values, but train on mrs02_36.
        :param train_ratio: the ratio for training data.
        :param valid_ratio: the ratio for validation data.
        :param test_ratio: the ratio for test data.
        :param random_seed: random seed for Python, Numpy and Tensorflow, useful when you want to compare experiments
        with different parameters.
        :param centre_test_set: use a centre as the test set, e.g. ChCh.
        :param pre_process_std_scale: when pre-processing, standard scale continuous data.
        :param pre_process_fill_missing_scans: when pre-processing, fill missing scans.
        :param pre_process_missing_scans_class: when pre-processing, what class to use when filling missing scans.
        :param prediction_path: the path to predicted outputs.
        :param features: features to select for the clinical model.
        """

        self.csv_path = csv_path
        self.scan_folder = scan_folder
        self.cache_folder = cache_folder
        self.target_class = target_class
        self.ct_key = ct_key
        self.ct_mask_key = ct_mask_key
        self.cta_key = cta_key

        # If stratify class is None then stratify the splits on the target class
        self.stratify_class = stratify_class
        if stratify_class is None:
            self.stratify_class = target_class

        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.centre_test_set = centre_test_set
        self.pre_process_std_scale = pre_process_std_scale
        self.pre_process_fill_missing_scans = pre_process_fill_missing_scans
        self.pre_process_missing_scans_class = pre_process_missing_scans_class
        self.prediction_path = prediction_path
        self.features = features
        self.target_features = {self.target_class, self.stratify_class}
        self.df_train, self.df_valid, self.df_test = self._load_and_impute()
        self.df = pd.concat([self.df_train, self.df_valid, self.df_test])
        self.df.reset_index(inplace=True, drop=True)

    def save_as_csv(self, file_path: str):
        self.df.to_csv(file_path, index=False)

    def num_classes(self, col_name: str):
        return self.df[col_name].nunique()

    @property
    def class_names(self):
        label_key = f"{self.target_class}_label"
        stats = self.df.groupby([self.target_class]).agg(
            {self.target_class: "first", label_key: "first"},
            index=False,
        )
        stats = stats.reset_index(inplace=False, drop=True)
        return stats.sort_values(by=[self.target_class])[label_key].tolist()

    def print_train_summary(self):
        self._print_summary(self.df_train, "train")

    def print_valid_summary(self):
        self._print_summary(self.df_valid, "valid")

    def print_test_summary(self):
        self._print_summary(self.df_test, "test")

    def _print_summary(self, df: pd.DataFrame, subset: str):
        target_class = self.target_class
        num_instances = len(df)
        num_classes = df[self.target_class].nunique()
        summary = (
            f"DeepMT Dataset Subset: {subset}:\n"
            f"    Target class: {target_class}:\n"
            f"    Num instances: {num_instances}\n"
            f"    Num classes: {num_classes}\n"
            f"    class_id, name, count, class %\n"
        )

        # Project level aggregation
        stats = df.groupby([target_class]).agg(
            {target_class: ["count"], f"{target_class}_label": "first"},
            index=False,
        )
        stats.reset_index(inplace=True)
        stats.columns = ["class_id", "count", "label"]
        for i, row in stats.iterrows():
            class_id = row["class_id"]
            label = row["label"]
            count = row["count"]
            ratio = count / num_instances * 100.0
            summary += f"    {class_id}, {label}, {count}, {ratio:.1f}%\n"

        count = 0
        summary += f"    Top cases:\n"
        for i, row in df[df["subset"] == subset].iterrows():
            summary += f"        {row['case_id']}\n"
            if count > 5:
                break
            count += 1

        print(summary)

    def pytorch_train(self, transform: Compose) -> Dataset:
        return self._make_pytorch(self.df_train, transform)

    def pytorch_valid(self, transform: Compose) -> Dataset:
        return self._make_pytorch(self.df_valid, transform)

    def pytorch_test(self, transform: Compose) -> Dataset:
        return self._make_pytorch(self.df_test, transform)

    def clinical_features(self, df: pd.DataFrame):
        clinical_feature_names = []
        columns_to_drop = set(self.TABULAR_X_DROP + list(self.target_features))
        for column_name in df.columns.tolist():
            if column_name not in columns_to_drop and (
                self.features is None or len(self.features) == 0 or column_name in self.features
            ):
                clinical_feature_names.append(column_name)

        return clinical_feature_names

    @property
    def n_clinical_features(self):
        return len(self.clinical_features(self.df_train))

    @property
    def clinical_feature_names(self):
        return self.clinical_features(self.df_train)

    def _make_pytorch(self, df: pd.DataFrame, transform: Compose) -> Dataset:
        # Make sure that ct_key and ct_mask_key are defined
        assert self.ct_key is not None, f"DeepMTDataset._make_pytorch: ct_key must be specified"
        assert self.ct_mask_key is not None, f"DeepMTDataset._make_pytorch: ct_mask_key must be specified"

        # Get clinical feature columns
        clinical_feature_names = self.clinical_features(df)

        # Create PyTorch dataset
        data = []
        for i, row in df.iterrows():
            ct_image = row["ct_image"]
            ct_mask = row["ct_mask"]
            cta_image = row["cta_image"]
            ct_exists = row["ct_exists"]
            cta_exists = row["cta_exists"]
            target_class = int(row[self.target_class])

            # Put into numpy array of shape (1, n_features) so that PyTorch collate function re-arranges the data
            # into a shape of (batch_size, 1, n_features)
            clinical_features = np.array([row[clinical_feature_names].tolist()], dtype=np.float32)

            item = {
                "case_id": row["case_id"],
                "ct_exists": ct_exists,
                "cta_exists": cta_exists,
                self.target_class: target_class,
                "ct_mask": ct_mask,
                "clinical_features": clinical_features,
            }

            # Only add image if it exists so that it can be filled in by a transform
            if ct_exists:
                item["ct_image"] = ct_image
            if self.cta_key is not None and cta_exists:
                item["cta_image"] = cta_image

            data.append(item)

        if self.cache_folder is not None:
            return PersistentDataset(data=data, transform=transform, cache_dir=self.cache_folder)

        return Dataset(data=data, transform=transform)

    def case_ids(self):
        return self.df_train["case_id"].tolist() + self.df_valid["case_id"].tolist() + self.df_test["case_id"].tolist()

    def index_from_case_id(self, subset: str, case_id: str):
        df = None
        if subset == "train":
            df = self.df_train
        elif subset == "valid":
            df = self.df_valid
        elif subset == "test":
            df = self.df_test
        else:
            raise ValueError("Invalid subset")

        # Get the index for case id
        matching_indices = df.index[df['case_id'] == case_id].tolist()
        if len(matching_indices) != 1:
            raise IndexError("Multiple matches")

        return matching_indices[0]

    def subsets(self):
        return self.df_train["subset"].tolist() + self.df_valid["subset"].tolist() + self.df_test["subset"].tolist()

    def tabular_train(self) -> Tuple[pd.DataFrame, np.ndarray]:
        return self._make_tabular(self.df_train)

    def tabular_valid(self) -> Tuple[pd.DataFrame, np.ndarray]:
        return self._make_tabular(self.df_valid)

    def tabular_test(self) -> Tuple[pd.DataFrame, np.ndarray]:
        return self._make_tabular(self.df_test)

    def _make_tabular(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        # From X drop all columns in TABULAR_X_DROP and all possible target features
        drop = set(self.TABULAR_X_DROP + list(self.target_features))
        X = df.drop(columns=drop, errors="ignore")
        y = df[self.target_class].to_numpy().astype(float)
        return X, y

    def _load_and_impute(self) -> TrainValidTestSplit:
        """Load and impute the dataset.

        :return: None.
        """

        # Set random seed
        logging.basicConfig(level=logging.DEBUG)

        # Load from CSV
        df = pd.read_csv(self.csv_path)

        # Original length of dataset
        num_cases_orig = df.shape[0]

        # Create scan paths
        ct_images = []
        cta_images = []
        ct_masks = []
        for index, row in df.iterrows():
            case_id = row["case_id"]
            centre = row["centre"]
            ct_path = None
            ct_mask_path = None
            cta_path = None

            if self.ct_key is not None:
                ct_path = os.path.normpath(
                    os.path.join(self.scan_folder, centre, case_id, "ax_CT", self.ct_key.format(case_id=case_id))
                )
                ct_mask_path = os.path.normpath(
                    os.path.join(self.scan_folder, centre, case_id, "ax_CT", self.ct_mask_key.format(case_id=case_id))
                )

            if self.cta_key is not None:
                cta_path = os.path.normpath(
                    os.path.join(self.scan_folder, centre, case_id, "ax_A", self.cta_key.format(case_id=case_id))
                )

            # Check that expected CTs and CTAs exist or not
            def validate_scan(path: str, expected):
                if os.path.isfile(path) and not expected:
                    logging.warning(f"Scan exists but not expected: {path}")
                elif not os.path.isfile(path) and expected:
                    logging.warning(f"Scan expected but doesn't exist: {path}")

            ct_exists = row["ct_exists"]
            cta_exists = row["cta_exists"]

            if ct_path is not None:
                validate_scan(ct_path, ct_exists)

            if cta_path is not None:
                validate_scan(cta_path, cta_exists)

            if ct_mask_path is not None and not os.path.isfile(ct_mask_path):
                logging.warning(f"Mask does not exist: {ct_mask_path}")

            ct_images.append(ct_path)
            ct_masks.append(ct_mask_path)
            cta_images.append(cta_path)

        df["ct_image"] = ct_images
        df["ct_mask"] = ct_masks
        df["cta_image"] = cta_images

        # Drop scans that don't exist
        df["ct_exists"] = df["ct_exists"].astype("bool")
        df["cta_exists"] = df["cta_exists"].astype("bool")

        # Strip strings, lower and replace spaces with underscores
        df_str = df.select_dtypes(["object"])
        df[df_str.columns] = df_str.apply(lambda v: v.str.strip())
        df_str = df.select_dtypes(["object"])
        df[df_str.columns] = df_str.apply(lambda v: v.str.replace(" ", "_"))

        # Quantize mRS
        df = self._quantize_mrs(df)
        df = self._quantize_age(df)

        # Drop cases with null target_class values
        df.dropna(subset=[self.stratify_class], inplace=True)

        # Drop cases which don't have a CT scan
        df = df[df["ct_exists"]]

        # Print summary
        num_cases = df.shape[0]
        num_cases_dropped = num_cases_orig - num_cases
        print(f"Num cases: {num_cases}, cases dropped: {num_cases_dropped}")
        print(f"Columns with na: {df.columns[df.isna().any()].tolist()}")

        # Add label column for target class (if not mRS)
        label_key = f"{self.target_class}_label"
        if label_key not in df:
            df[label_key] = df[self.target_class].astype(str)
            self.target_features.add(label_key)

        # Split data into train, valid and test sets
        if self.centre_test_set is not None:
            df_train, df_valid, df_test = centre_split(
                df=df,
                centre_test_set=self.centre_test_set,
                valid_ratio=self.valid_ratio,
                stratify_class=self.stratify_class,
                random_seed=self.random_seed,
            )
        else:
            df_train, df_valid, df_test = standard_split(
                df=df,
                train_ratio=self.train_ratio,
                valid_ratio=self.valid_ratio,
                test_ratio=self.test_ratio,
                stratify_class=self.stratify_class,
                random_seed=self.random_seed,
            )
            # df_train, df_valid, df_test = standard_split2(
            #     df=df,
            #     train_ratio=self.train_ratio,
            #     valid_ratio=self.valid_ratio,
            #     test_ratio=self.test_ratio,
            #     stratify_class=self.stratify_class,
            #     random_seed=self.random_seed,
            # )

        # Fill missing scans
        # Do before values are one hot encoded
        if self.pre_process_fill_missing_scans:
            df_train, df_valid, df_test = fill_missing_scans(
                target_class=self.pre_process_missing_scans_class, df_train=df_train, df_valid=df_valid, df_test=df_test
            )

        # Add predicted imaging_score values from an imaging model
        if self.prediction_path is not None:
            df_train, df_valid, df_test = load_predictions(
                prediction_path=self.prediction_path, df_train=df_train, df_valid=df_valid, df_test=df_test
            )

        # Impute missing data
        preprocess = make_pre_processing_pipeline(standard_scale=self.pre_process_std_scale)
        train_transformed = preprocess.fit_transform(df_train)
        feature_names_out = [
            name.split("__")[1] for name in preprocess.get_feature_names_out(df_train.columns.tolist())
        ]
        df_train = pd.DataFrame(train_transformed, columns=feature_names_out)
        df_valid = pd.DataFrame(preprocess.transform(df_valid), columns=feature_names_out)
        df_test = pd.DataFrame(preprocess.transform(df_test), columns=feature_names_out)

        # Add subsets
        df_train["subset"] = "train"
        df_valid["subset"] = "valid"
        df_test["subset"] = "test"

        # Reset index: so that indexes are incremental
        df_train.reset_index(inplace=True, drop=True)
        df_valid.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)

        return df_train, df_valid, df_test

    def _quantize_mrs(self, df: pd.DataFrame) -> pd.DataFrame:
        # Quantize
        for member in mRSType:
            df[member.value] = df["mrs"].apply(lambda mrs: mrs_quantize(mrs, member))
            label_key = f"{member.value}_label"
            df[label_key] = df[member.value].apply(lambda code: mRSType.class_name(member, code))
            self.target_features.add(member.value)

        return df

    def _quantize_age(self, df: pd.DataFrame) -> pd.DataFrame:
        age_key = "age"
        age_median_key = "age_median"
        age_median = int(df[age_key].median())
        age_min = int(df[age_key].min())
        age_max = int(df[age_key].max())
        df[age_median_key] = df[age_key].apply(lambda age: int(age < age_median))
        label_key = f"{age_median_key}_label"
        df[label_key] = df[age_median_key].apply(
            lambda age: f"age {age_min}-{age_median-1}" if age == 0 else f"age {age_median}-{age_max}"
        )
        self.target_features.add(age_median_key)

        return df
