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
import glob
import logging
import os
import sys
from timeit import default_timer as timer
from typing import Dict, List

import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
import torch
from natsort.natsort import natsorted
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from deep_mt.charts import plot_confusion_matrix
from deep_mt.config import Config
from deep_mt.dataset import DeepMTDataset
from deep_mt.ml_utils import make_model, make_transforms
from deep_mt.monai_utils import set_determinism


def make_case_report(case_ids: List, y_true: List, y_pred: List, class_names: List):
    results = []
    for case_id, y_true_, y_pred_ in zip(case_ids, y_true, y_pred):
        correct = y_true_ == y_pred_
        y_true_label = class_names[y_true_]
        y_pred_label = class_names[y_pred_]
        results.append({"correct": correct, "result": f"{y_true_label} -> {y_pred_label}", "case_id": case_id})

    results.sort(key=lambda x: (x["correct"], x["result"], x["case_id"]))

    return results


def display_case_report(experiment_folder: str, data: List[Dict], df: pd.DataFrame = None):
    for record in data:
        experiment_name = record["experiment_name"]
        run_name = record["weights"].replace(".pth", "")
        case_ids = record["case_ids"]
        y_true = record["y_true"]
        y_pred_class_ids = record["y_pred_class_ids"]
        class_names = record["class_names"]

        results = make_case_report(case_ids, y_true, y_pred_class_ids, class_names)
        df_results = pd.DataFrame(results)

        if df is not None:
            df_results = pd.merge(df_results, df, on="case_id", how="left")

        output_path = os.path.join(experiment_folder, f"cases_{run_name}.csv")
        df_results.to_csv(output_path, index=False)


def evaluate(config: Config, subset: str):
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

    # Create common variables
    pin_memory = torch.cuda.is_available()

    # Make class names
    n_classes = dataset.num_classes(config.target_class)
    class_names = dataset.class_names

    # Define transforms
    # Use valid transforms for evaluation as random transforms should not be applied when evaluating
    _, trans_valid = make_transforms(config)

    # Create a test data loader
    if subset == "train":
        ds = dataset.pytorch_train(trans_valid)
    elif subset == "valid":
        ds = dataset.pytorch_valid(trans_valid)
    else:
        ds = dataset.pytorch_test(trans_valid)

    data_loader = DataLoader(ds, batch_size=config.batch_size, num_workers=config.n_workers, pin_memory=pin_memory)

    # List all weights in directory
    weights_files = glob.glob(os.path.join(config.experiment_folder, "*.pth"))
    weights_files = natsorted(weights_files)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate models
    data = []
    for weights_file in weights_files:
        print(f"Evaluating weights file: {weights_file}")

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

            # Save results
            y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
            y_softmax = softmax(y_pred, axis=1)
            y_pred_class_ids = np.argmax(y_pred, axis=1)

            # Compute scores
            accuracy = metrics.accuracy_score(y_true, y_pred_class_ids)
            balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred_class_ids)

            roc_auc = np.nan
            if n_classes <= 2:
                y_score = y_softmax[:, 1]
                roc_auc = metrics.roc_auc_score(y_true, y_score)

            f1score = metrics.f1_score(y_true, y_pred_class_ids, average="macro")
            precision = metrics.precision_score(y_true, y_pred_class_ids, average="macro")
            recall = metrics.recall_score(y_true, y_pred_class_ids, average="macro")

            # Add to results
            data.append(
                {
                    "experiment_name": config.experiment_name,
                    "weights": os.path.basename(weights_file),
                    "case_ids": case_ids,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_pred_class_ids": y_pred_class_ids,
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_accuracy,
                    "roc_auc": roc_auc,
                    "f1score": f1score,
                    "precision": precision,
                    "recall": recall,
                    "weights_file": os.path.normpath(weights_file),
                    "class_names": class_names,
                }
            )

    # Save results
    subset_folder = os.path.join(config.experiment_folder, subset)
    os.makedirs(subset_folder, exist_ok=True)
    output_path = os.path.join(subset_folder, "evaluation-results.csv")
    df = pd.DataFrame(data=data)
    df.to_csv(
        output_path,
        columns=[
            "experiment_name",
            "weights",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "f1score",
            "precision",
            "recall",
            "weights_file",
        ],
        index=False,
    )
    print(f"Saved results to: {output_path}")

    # Save case data
    display_case_report(subset_folder, data, df=dataset.df)

    # Display evaluation charts etc
    display_evaluation(subset_folder, data)

    # Print duration
    end = timer()
    duration = end - start
    print(f"Duration: {datetime.timedelta(seconds=duration)}")


def display_evaluation(experiment_folder: str, data: List[Dict]):
    for record in data:
        experiment_name = record["experiment_name"]
        run_name = record["weights"].replace(".pth", "")
        y_true = record["y_true"]
        y_pred_class_ids = record["y_pred_class_ids"]
        class_names = record["class_names"]

        # Print classification reports
        report = classification_report(y_true, y_pred_class_ids)
        print(f"Classification Report: {run_name}")
        print(report)

        # Print confusion matrices
        normalize = True
        plot_confusion_matrix(
            y_true,
            y_pred_class_ids,
            classes=class_names,
            normalize=normalize,
            title=f"Confusion Matrix: {run_name}",
        )
        figure_path = os.path.join(experiment_folder, f"conf_mat_{run_name}.png")
        plt.savefig(figure_path)
