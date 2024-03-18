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
import datetime
import logging
import os
import sys
from timeit import default_timer as timer

import monai
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader

from deep_mt.config import Config
from deep_mt.dataset import DeepMTDataset
from deep_mt.ml_utils import make_model, make_transforms
from deep_mt.monai_utils import set_determinism


def predict(config: Config, weights_file: str):
    start = timer()
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_determinism(seed=config.random_seed, use_deterministic_algorithms=True, warn_only=True)

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
        features=config.features,
    )

    # Print dataset summary
    dataset.print_train_summary()
    dataset.print_valid_summary()
    dataset.print_test_summary()

    # Define transforms
    # Use valid transforms for evaluation as random transforms should not be applied when evaluating
    _, trans_valid = make_transforms(config)

    # Evaluate models
    data = []
    for subset in ["train", "valid", "test"]:
        print(f"Predicting for {subset} subset and weights file: {weights_file}")

        # Create a test data loader
        if subset == "train":
            ds = dataset.pytorch_train(trans_valid)
        elif subset == "valid":
            ds = dataset.pytorch_valid(trans_valid)
        else:
            ds = dataset.pytorch_test(trans_valid)
        data_loader = DataLoader(
            ds, batch_size=config.batch_size, num_workers=config.n_workers, pin_memory=torch.cuda.is_available()
        )

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load & eval model
        model_kwargs = copy.copy(config.model_kwargs)
        if config.model_name == "deep_mt.hybrid_model.Densenet121TabularHybrid":
            model_kwargs["n_tab_features"] = dataset.n_clinical_features
            print(f"model_kwargs: {model_kwargs}")
            print(f"n_clinical_features: {dataset.n_clinical_features}")
            print(f"clinical_features: {dataset.clinical_features(dataset.df_train)}")

        model = make_model(config.model_name, model_kwargs)
        model.load_state_dict(torch.load(weights_file))
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            case_ids, y_true, y_pred = list(), list(), list()
            for batch in data_loader:
                images, labels = batch[config.output_key].to(device), batch[config.target_class].to(device)

                if config.model_name == "deep_mt.hybrid_model.Densenet121TabularHybrid":
                    tab_data = batch["clinical_features"].to(device)
                    outputs = model(images, tab_data)
                else:
                    outputs = model(images)

                # Add to predictions and true vals
                y_true.append(labels.cpu().detach().numpy())
                y_pred.append(outputs.cpu().detach().numpy())
                case_ids += batch["case_id"]

            # Get results
            y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
            y_softmax = softmax(y_pred, axis=1)
            y_score = y_softmax[:, 1]
            y_pred_class_ids = np.argmax(y_pred, axis=1)

            # Combine into list of tuples
            data += list(
                zip(
                    case_ids,
                    [subset] * len(case_ids),
                    y_pred_class_ids,
                    y_score,
                )
            )

    # Save results
    output_path = os.path.join(config.experiment_folder, f"{config.experiment_name}-predictions.csv")
    df = pd.DataFrame(
        data=data,
        columns=[
            "case_id",
            "subset",
            "class_id",
            "score",
        ],
    )
    df.to_csv(
        output_path,
        index=False,
    )
    print(f"Saved results to: {output_path}")

    # Print duration
    end = timer()
    duration = end - start
    print(f"Duration: {datetime.timedelta(seconds=duration)}")
