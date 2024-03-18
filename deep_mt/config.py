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
import os
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Dict, Tuple, List, Optional

import yaml
from cerberus import Validator

DICOM_EXTENSION = ".dcm"


@dataclass
class MissingScans:
    fill: bool
    class_name: Optional[str]

    @classmethod
    def from_dict(cls, dict_: Dict) -> MissingScans:
        return MissingScans(dict_.get("fill"), dict_.get("class_name"))


@dataclass
class Config:
    # Paths to data
    experiment_name: str
    csv_path: str
    scan_folder: str
    experiments_folder: str

    # Target class we are training and testing for
    target_class: str

    ct_key: str = None
    cta_key: str = None
    ct_mask_key: str = None
    resume: bool = None

    stratify_class: str = None
    output_key: str = "ct_cta_image"

    input_shape: Tuple[int, int, int] = None
    output_shape: Tuple[int, int, int] = (96, 96, 96)
    transform_name: str = "baseline"
    transform_kwargs: Dict = field(default_factory=dict)
    in_channels: int = 1

    # Pre-processing
    std_scale: bool = True
    missing_scans: MissingScans = field(default_factory=lambda: MissingScans(False, None))

    # Model parameters
    model_name: str = "monai.networks.nets.DenseNet121"
    model_kwargs: Dict = field(default_factory=dict)
    pretrained_weights_path: str = None

    # Loss function
    loss_name: str = "torch.nn.CrossEntropyLoss"
    loss_kwargs: Dict = field(default_factory=dict)
    class_weights: bool = True

    # Optimiser
    optimiser_name: str = "torch.optim.Adam"
    optimiser_kwargs: Dict = field(default_factory=dict)

    features: List = field(default_factory=list)

    # How to split the dataset
    train_ratio: float = 0.7
    valid_ratio: float = 0.1
    test_ratio: float = 0.2
    centre_test_set: str = None

    # Random seed for repeatable experiments
    random_seed: int = None
    n_workers: int = cpu_count()
    n_epochs: int = 10
    batch_size: int = 10

    # Transfer learning
    fine_tune: bool = False
    weights_path: str = None

    # Path to the predicted results
    prediction_path: str = None

    # Image weight in multi task loss. loss = img_weight*img_loss + (1-img_weight)*tab_loss
    image_weight: float = 1.0

    @property
    def experiment_folder(self) -> str:
        folder = os.path.join(self.experiments_folder, self.experiment_name)
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def cache_folder(self) -> str:
        folder = os.path.join(self.experiment_folder, "cache")
        os.makedirs(folder, exist_ok=True)
        return folder

    @classmethod
    def load(cls, path: str):
        """Load the config file.

        :return: the Config instance.
        """

        # Open file
        with open(path, "r") as f:
            dict_ = yaml.safe_load(f)

        # Validate config
        schema = cls.make_schema()
        v = Validator(schema)
        is_valid = v.validate(dict_)

        # Check if valid
        if not is_valid:
            msg = f"Config file invalid: {path}"
            logging.error(msg)
            for key, values in v.errors.items():
                logging.error(f"{key}: {values}")
            raise Exception(msg)

        # Parse config from dictionary
        root_folder = os.path.dirname(path)

        return cls.from_dict(dict_, root_folder)

    @staticmethod
    def make_schema():
        return {
            "experiment_name": {"required": True, "type": "string"},
            "csv_path": {"required": True, "type": "string"},
            "scan_folder": {"required": True, "type": "string"},
            "experiments_folder": {"required": True, "type": "string"},
            "ct_key": {"required": False, "type": "string"},
            "cta_key": {"required": False, "type": "string"},
            "ct_mask_key": {"dependencies": ["ct_key"], "type": "string"},
            "output_key": {"required": False, "type": "string"},
            "target_class": {"required": True, "type": "string"},
            "stratify_class": {"required": False, "type": "string"},
            "transform_name": {"required": False, "type": "string"},
            "transform_kwargs": {"required": False, "type": "dict"},
            "in_channels": {"required": False, "type": "integer"},
            "std_scale": {"required": False, "type": "boolean"},
            "missing_scans": {
                "required": True,
                "type": "dict",
                "schema": {
                    "fill": {"required": True, "type": "boolean"},
                    "class_name": {"required": False, "type": "string"},
                },
            },
            "input_shape": {
                "required": False,
                "type": "list",
                "items": [{"type": "integer"}, {"type": "integer"}, {"type": "integer"}],
            },
            "output_shape": {
                "required": False,
                "type": "list",
                "items": [{"type": "integer"}, {"type": "integer"}, {"type": "integer"}],
            },
            "model_name": {"required": False, "type": "string"},
            "model_kwargs": {"required": False, "type": "dict"},
            "loss_name": {"required": False, "type": "string"},
            "loss_kwargs": {"required": False, "type": "dict"},
            "class_weights": {"required": False, "type": "boolean"},
            "optimiser_name": {"required": False, "type": "string"},
            "optimiser_kwargs": {"required": False, "type": "dict"},
            "features": {"required": False, "type": "list"},
            "train_ratio": {"required": False, "type": "float", "min": 0.0, "max": 1.0},
            "valid_ratio": {"required": False, "type": "float", "min": 0.0, "max": 1.0},
            "test_ratio": {"required": False, "type": "float", "min": 0.0, "max": 1.0},
            "centre_test_set": {"required": False, "type": "string"},
            "random_seed": {"required": False, "type": "integer"},
            "n_workers": {"required": False, "type": "integer"},
            "n_epochs": {"required": False, "type": "integer"},
            "batch_size": {"required": False, "type": "integer"},
            "fine_tune": {"required": False, "type": "boolean"},
            "weights_path": {"required": False, "type": "string"},
            "resume": {"required": False, "type": "boolean"},
            "prediction_path": {"required": False, "type": "string"},
            "image_weight": {"required": False, "type": "float", "min": 0.0, "max": 1.0},
            "pretrained_weights_path": {"required": False, "type": "string"},
        }

    @classmethod
    def from_dict(cls, dict_: Dict, root_folder: str) -> Config:
        """Convert Dict into Config.

        :param dict_: the dictionary.
        :param root_folder: the root folder containing all the data files.
        :return: the Config instance.
        """

        # Paths to data
        experiment_name = dict_.get("experiment_name")
        csv_path = os.path.normpath(os.path.join(root_folder, dict_.get("csv_path")))
        scan_folder = os.path.normpath(os.path.join(root_folder, dict_.get("scan_folder")))
        experiments_folder = os.path.normpath(os.path.join(root_folder, dict_.get("experiments_folder")))
        ct_key = dict_.get("ct_key", Config.ct_key)
        cta_key = dict_.get("cta_key", Config.cta_key)
        ct_mask_key = dict_.get("ct_mask_key", Config.ct_mask_key)
        output_key = dict_.get("output_key")
        pretrained_weights_path = dict_.get("pretrained_weights_path", Config.pretrained_weights_path)
        if pretrained_weights_path is not None:
            pretrained_weights_path = os.path.normpath(os.path.join(root_folder, pretrained_weights_path))

        # Target class we are training and testing for
        target_class = dict_.get("target_class")
        stratify_class = dict_.get("stratify_class", Config.stratify_class)

        input_shape = dict_.get("input_shape", Config.input_shape)
        if input_shape is not None:
            input_shape = tuple(input_shape)
        output_shape = tuple(dict_.get("output_shape", Config.output_shape))
        transform_name = dict_.get("transform_name", Config.transform_name)
        transform_kwargs = dict_.get("transform_kwargs", dict())
        in_channels = dict_.get("in_channels", Config.in_channels)

        # Pre-processing
        std_scale = dict_.get("std_scale", Config.std_scale)
        missing_scans = MissingScans.from_dict(dict_.get("missing_scans"))

        # Model parameters
        model_name = dict_.get("model_name", Config.model_name)
        model_kwargs = dict_.get("model_kwargs", dict())
        resume = dict_.get("resume", None)

        # Loss
        loss_name = dict_.get("loss_name", Config.model_name)
        loss_kwargs = dict_.get("loss_kwargs", dict())
        class_weights = dict_.get("class_weights", Config.class_weights)

        # Optimiser
        optimiser_name = dict_.get("optimiser_name", Config.model_name)
        optimiser_kwargs = dict_.get("optimiser_kwargs", dict())

        features = dict_.get("features", list())

        # How to split the dataset
        train_ratio = dict_.get("train_ratio", Config.train_ratio)
        valid_ratio = dict_.get("valid_ratio", Config.valid_ratio)
        test_ratio = dict_.get("test_ratio", Config.test_ratio)
        centre_test_set = dict_.get("centre_test_set", Config.centre_test_set)

        # Random seed for repeatable experiments
        random_seed = dict_.get("random_seed", Config.random_seed)
        n_workers = dict_.get("n_workers", Config.n_workers)
        n_epochs = dict_.get("n_epochs", Config.n_epochs)
        batch_size = dict_.get("batch_size", Config.batch_size)

        # Transfer learning
        fine_tune = dict_.get("fine_tune", Config.fine_tune)
        weights_path = dict_.get("weights_path", Config.weights_path)

        # Predicted results
        prediction_path = dict_.get("prediction_path", Config.prediction_path)
        if prediction_path is not None:
            prediction_path = os.path.normpath(os.path.join(root_folder, prediction_path))

        # Image weight
        image_weight = dict_.get("image_weight", 1.0)

        # Custom validation
        assert not (
            cta_key is None and transform_kwargs.get("cta", False)
        ), "transform_kwargs has cta: true, so please specify a value for cta_key"

        return Config(
            experiment_name=experiment_name,
            csv_path=csv_path,
            scan_folder=scan_folder,
            experiments_folder=experiments_folder,
            target_class=target_class,
            ct_key=ct_key,
            cta_key=cta_key,
            ct_mask_key=ct_mask_key,
            output_key=output_key,
            stratify_class=stratify_class,
            input_shape=input_shape,
            output_shape=output_shape,
            transform_name=transform_name,
            transform_kwargs=transform_kwargs,
            in_channels=in_channels,
            std_scale=std_scale,
            missing_scans=missing_scans,
            model_name=model_name,
            model_kwargs=model_kwargs,
            loss_name=loss_name,
            loss_kwargs=loss_kwargs,
            class_weights=class_weights,
            optimiser_name=optimiser_name,
            optimiser_kwargs=optimiser_kwargs,
            features=features,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            centre_test_set=centre_test_set,
            random_seed=random_seed,
            n_workers=n_workers,
            n_epochs=n_epochs,
            batch_size=batch_size,
            fine_tune=fine_tune,
            weights_path=weights_path,
            resume=resume,
            prediction_path=prediction_path,
            image_weight=image_weight,
            pretrained_weights_path=pretrained_weights_path,
        )
