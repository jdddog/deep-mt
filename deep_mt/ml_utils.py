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

import copy
import math
import os
import random
from dataclasses import dataclass
from operator import itemgetter
from pydoc import locate
from typing import Tuple, List, Type, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from deep_mt.config import Config


def set_random(seed: int):
    """Set random seed for Python, Numpy and Pytorch.

    :param seed: the seed value to set.
    :return: None.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_sample(X, y, sample_size, random_seed):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=random_seed)
    sample_index, out_index = next(sss.split(X, y))
    return sample_index, out_index


def stratified_split(y, random_seed, test_size=0.1):
    X = np.zeros(len(y))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    sss.get_n_splits(X, y)
    train_indicies, test_indicies = next(sss.split(X, y))
    return train_indicies, test_indicies


def shuffle_split(n: int, random_seed, test_size=0.1):
    X = np.zeros(n)
    ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_indicies, test_indicies = next(ss.split(X))
    return train_indicies, test_indicies


def get_max_weights(path: str):
    weight_files = [f for f in os.listdir(path) if f.startswith("weights") and f.endswith(".hdf5")]
    file_name_max = ""
    val_accuracy_max = 0.0
    for file_name in weight_files:
        values = file_name.replace(".hdf5", "").split("-")
        val_accuracy = float(values[2])
        if val_accuracy_max < val_accuracy:
            val_accuracy_max = val_accuracy
            file_name_max = file_name
    return file_name_max


def split_largest_remainder(sample_size: int, *ratios) -> Tuple:
    """Split a sample size into different groups based on a list of ratios (that add to 1.0) using the largest
    remainder method: https://en.wikipedia.org/wiki/Largest_remainder_method.

    :param sample_size: the absolute sample size.
    :param ratios: the list of ratios, must add to 1.0.
    :return: the absolute numbers of each group.
    """

    assert sum(ratios) == 1, "ratios must sum to 1.0"
    sizes = [sample_size * ratio for ratio in ratios]
    sizes_whole = [math.floor(size) for size in sizes]

    while (sample_size - sum(sizes_whole)) > 0:
        remainders = [size % 1 for size in sizes]
        max_index = max(enumerate(remainders), key=itemgetter(1))[0]
        sizes_whole[max_index] = sizes_whole[max_index] + 1
        sizes[max_index] = sizes_whole[max_index]

    return tuple(sizes_whole)


@dataclass
class Impute:
    name: str = None
    type: Type = None
    fill_strategy: str = None


def calc_missing_values(df, impute_strategy: List[Impute]):
    values = []

    for impute in impute_strategy:
        value = None
        if impute.fill_strategy == "mean":
            value = df[impute.name].mean()
        elif impute.fill_strategy == "mode":
            value = df[impute.name].mode().iloc[0]
        elif impute.fill_strategy == "median":
            value = df[impute.name].median()

        value = impute.type(value)
        values.append(value)

    return values


def fill_missing_values(df, impute_strategy: List[Impute], values, debug=False):
    df_copy = df.copy(deep=True)

    for impute, value in zip(impute_strategy, values):
        df_copy[impute.name] = df_copy[impute.name].fillna(value)

    if debug:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print("***fill_missing_values***")
            print(df_copy)

    return df_copy


def make_transforms(config: Config):
    """Make a transform based on a class name and kwargs.

    Note that, confusingly, ct_key and cta_key are different variables than in the config file.

    :param config: the config file.
    :return: the transform instance.
    """

    func = locate(config.transform_name)
    func_kwargs = {
        **{
            "input_shape": config.input_shape,
            "output_shape": config.output_shape,
            "output_key": config.output_key,
        },
        **config.transform_kwargs,
    }
    return func(**func_kwargs)


def make_model(class_name: str, kwargs: Dict):
    """Make a model based on a class name and kwargs.

    :param class_name: the full Python module path to the model class.
    :param kwargs: the kwargs to pass to the model class constructor.
    :return: the model instance.
    """

    cls = locate(class_name)
    return cls(**kwargs)


def make_loss(class_name: str, kwargs: Dict, weight=None):
    """Make a loss function based on a class name and kwargs.

    :param class_name: the full Python module path to the loss class.
    :param kwargs: the kwargs to pass to the class constructor.
    :return: the model instance.
    """

    # Add optional class weights
    kwargs_ = copy.deepcopy(kwargs)
    if weight is not None:
        kwargs_["weight"] = weight

    cls = locate(class_name)
    return cls(**kwargs_)


def make_optimiser(class_name: str, params, kwargs: Dict):
    """Make an optimiser based on a class name and kwargs.

    :param class_name: the full Python module path to the class.
    :param kwargs: the kwargs to pass to the class constructor.
    :return: the model instance.
    """

    cls = locate(class_name)
    return cls(params, **kwargs)
