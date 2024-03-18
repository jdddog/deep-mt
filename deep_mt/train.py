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
import logging
import os
import sys
from datetime import datetime
from glob import glob
from typing import Union

import joblib
import monai
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader
from monai.networks.utils import copy_model_state
from sklearn import clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import _safe_indexing
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter

from deep_mt.config import Config
from deep_mt.dataset import DeepMTDataset
from deep_mt.ml_utils import make_model, make_loss, make_optimiser, make_transforms
from deep_mt.monai_utils import set_determinism
from deep_mt.utils import print_duration

ONE_HUNDRED_PERCENT = 100.0


def get_latest_pth(experiment_folder: str) -> Union[str, None]:
    files = glob(os.path.join(experiment_folder, "*.pth"))

    if len(files) == 0:
        return None

    latest = max(files, key=lambda x: os.path.getctime(x))
    return latest


def get_epoch_from_pth(path) -> int:
    start = path.find("epoch_") + len("epoch_")
    end = len(path) - len(".pth")
    epoch = int(path[start:end])
    return epoch


@print_duration
def train_pytorch(config: Config):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_determinism(seed=config.random_seed, use_deterministic_algorithms=True, warn_only=True)

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

    # Save dataset state
    csv_output_path = os.path.normpath(os.path.join(config.experiment_folder, f"{config.experiment_name}.csv"))
    print(f"Saving dataset state to: {csv_output_path}")
    dataset.save_as_csv(csv_output_path)

    # Create common variables
    pin_memory = torch.cuda.is_available()

    # Define transforms
    trans_train, trans_valid = make_transforms(config)

    # Create a training data loader
    ds_train = dataset.pytorch_train(trans_train)
    # Use Monai DataLoader rather than torch DataLoader because it fixes a problem with random seeds across workers
    # See this link for more details: https://github.com/Project-MONAI/MONAI/issues/1068#issuecomment-699647848
    dl_train = DataLoader(
        ds_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.n_workers,
        pin_memory=pin_memory,
    )

    # Create a validation data loader
    ds_valid = dataset.pytorch_valid(trans_valid)
    dl_valid = DataLoader(
        ds_valid,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=config.n_workers,
        pin_memory=pin_memory,
    )

    # Create class weights
    class_weights = None
    if config.class_weights:
        y = np.array([row[config.target_class] for row in ds_train.data])
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        class_weights = torch.FloatTensor(weights).cuda()

    # Create model, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained = make_model("monai.networks.nets.DenseNet121", config.model_kwargs)
    pretrained.load_state_dict(torch.load(config.pretrained_weights_path))

    model_kwargs = copy.copy(config.model_kwargs)
    if config.model_name == "deep_mt.hybrid_model.Densenet121TabularHybrid":
        model_kwargs["n_tab_features"] = dataset.n_clinical_features
        print(f"model_kwargs: {model_kwargs}")
        print(f"n_clinical_features: {dataset.n_clinical_features}")
        print(f"clinical_features: {dataset.clinical_features(dataset.df_train)}")
    model = make_model(config.model_name, model_kwargs)

    epoch_offset = 0
    if config.resume:
        latest_pth = get_latest_pth(config.experiment_folder)
        if latest_pth:
            epoch_offset = get_epoch_from_pth(latest_pth)
            model.load_state_dict(torch.load(latest_pth))
            print(f"Resuming from {latest_pth} at epoch {epoch_offset}")

    # Load pre-trained parameters, updates model inplace
    pretrained_dict, updated_keys, unchanged_keys = copy_model_state(
        model, pretrained.features, dst_prefix="img_features.", inplace=True
    )

    # Freeze image features
    for param in model.img_features.parameters():
        param.requires_grad = False

    # Freeze class_layers except for reduced_dim
    for param in model.class_layers.parameters():
        param.requires_grad = True

    print("Weights")
    path = os.path.join(config.experiment_folder, f"{config.experiment_name}_test.txt")
    with open(path, mode="w") as f:
        for param in model.parameters():
            f.write(f"{param}")

    print("Model summary")
    print(model)

    # Setup transfer learning if applicable
    if config.fine_tune:
        weights_path = os.path.join(config.experiments_folder, config.weights_path)
        print(f"Fine tuning with weights: {weights_path}")
        model.load_state_dict(torch.load(weights_path))

        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze class layers
        for param in model.class_layers.parameters():
            param.requires_grad = True

    # Send model to device
    model = model.to(device)

    # Create loss and optimiser
    loss_function = make_loss(config.loss_name, config.loss_kwargs, weight=class_weights)
    optimizer = make_optimiser(config.optimiser_name, model.parameters(), config.optimiser_kwargs)

    # Start PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    log_dir = f"runs/{config.experiment_name}-{datetime.now().strftime('%b%d-%H-%M-%S')}"
    print(f"Tensorboard run: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(config.n_epochs):
        print("-" * 10)
        e_epoch = epoch + 1 + epoch_offset
        print(f"epoch {e_epoch}/{config.n_epochs + epoch_offset}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch in dl_train:
            step += 1
            images, labels = batch[config.output_key].to(device), batch[config.target_class].to(device)
            tab_data = batch["clinical_features"].to(device)

            optimizer.zero_grad()

            if config.model_name == "deep_mt.hybrid_model.Densenet121TabularHybrid":
                outputs = model(images, tab_data)
            else:
                outputs = model(images)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(ds_train) // dl_train.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * e_epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {e_epoch} average loss: {epoch_loss:.4f}")

        if (e_epoch) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                for batch in dl_valid:
                    images, labels = batch[config.output_key].to(device), batch[config.target_class].to(device)
                    tab_data = batch["clinical_features"].to(device)
                    # outputs, aux_outputs = model(images, tab_data)
                    outputs = model(images, tab_data)

                    value = torch.eq(outputs.argmax(dim=1), labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
                metric = num_correct / metric_count
                metric_values.append(metric)
                weights = os.path.join(config.experiment_folder, f"{config.experiment_name}_epoch_{e_epoch}.pth")
                torch.save(model.state_dict(), weights)
                print("saved new best metric model")
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = e_epoch
                print(
                    "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        e_epoch, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", metric, e_epoch)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


def evaluate(subset: str, y_true, y_pred, y_pred_score):
    acc = accuracy_score(y_true, y_pred) * ONE_HUNDRED_PERCENT
    y_pred_score = [v[1] for v in y_pred_score]
    auc = roc_auc_score(y_true, y_pred_score) * ONE_HUNDRED_PERCENT
    print(subset)
    print(f"  Accuracy: {acc:.2f}%")
    print(f"  ROC AUC: {auc:.2f}")
    print("")


@print_duration
def train_sklearn(config: Config):
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    set_determinism(seed=config.random_seed)

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
        prediction_path=config.prediction_path,
    )

    # Print dataset summary
    dataset.print_train_summary()
    dataset.print_valid_summary()
    dataset.print_test_summary()

    # Save dataset state
    csv_output_path = os.path.normpath(os.path.join(config.experiment_folder, f"{config.experiment_name}.csv"))
    print(f"Saving dataset state to: {csv_output_path}")
    dataset.save_as_csv(csv_output_path)

    # Train and evaluate
    X_train, y_train = dataset.tabular_train()
    X_valid, y_valid = dataset.tabular_valid()
    X_test, y_test = dataset.tabular_test()

    # Select subset of features if features has values
    if len(config.features):
        X_train = X_train[config.features]
        X_valid = X_valid[config.features]
        X_test = X_test[config.features]

    # Make model
    model = make_model(config.model_name, config.model_kwargs)
    model.fit(X_train, y_train)
    print(model)

    features_all = X_train.columns.tolist()
    results, y_preds, y_scores = sklearn_evaluate(
        model, X_train, y_train, X_valid, y_valid, X_test, y_test, config.experiment_name, features_all
    )

    # Save Results CSV
    columns = ["experiment", "train_acc", "train_auc", "valid_acc", "valid_auc", "test_acc", "test_auc", "features"]
    df = pd.DataFrame([results], columns=columns)
    file_path = os.path.join(config.experiment_folder, f"{config.experiment_name}.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved results to: {file_path}")

    # Save prediction CSV
    columns = ["case_id", "subset", "y_pred", "y_score"]
    case_ids = dataset.case_ids()
    subsets = dataset.subsets()
    data = list(zip(case_ids, subsets, y_preds, y_scores))
    df = pd.DataFrame(data, columns=columns)
    file_path = os.path.join(config.experiment_folder, f"{config.experiment_name}-predictions.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved predictions to: {file_path}")


@print_duration
def feature_selection(config: Config, n_splits: int, n_repeats: int):
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    set_determinism(seed=config.random_seed)

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

    X_train, y_train = dataset.tabular_train()
    X_valid, y_valid = dataset.tabular_valid()
    X_test, y_test = dataset.tabular_test()
    features_all = X_train.columns.tolist()
    print("Features:")
    for feature in features_all:
        print(feature)

    # Make model
    model = make_model(config.model_name, config.model_kwargs)

    # Select features
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=config.random_seed)
    wrapper = SequentialFeatureSelector(clone(model), cv=cv, n_features_to_select="auto", tol=1e-5, direction="forward")
    wrapper.fit(X_train, y_train)

    # Print features
    results = []
    for i, name in enumerate(features_all):
        selected = wrapper.support_[i]
        results.append((selected, name))
    results = sorted(results)
    features_subset = []
    for selected, name in results:
        if selected:
            features_subset.append(name)
            print(f"Feature: {name}, selected {selected}")

    print("All features")
    model = model.fit(X_train, y_train)

    file_path = os.path.join(config.experiment_folder, f"{config.experiment_name}_all_features.joblib")
    joblib.dump(model, file_path)

    # Evaluate model
    results = []
    result, y_preds, y_scores = sklearn_evaluate(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        f"{config.experiment_name}_all_features",
        features_all,
    )
    results.append(result)

    columns = ["case_id", "subset", "y_pred", "y_score"]
    case_ids = dataset.case_ids()
    subsets = dataset.subsets()
    data = list(zip(case_ids, subsets, y_preds, y_scores))
    df = pd.DataFrame(data, columns=columns)
    file_path = os.path.join(config.experiment_folder, f"{config.experiment_name}-predictions.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved predictions to: {file_path}")

    # Results subset
    print(f"Feature subset:")
    feature_subset = wrapper.get_support()
    X_train = _safe_indexing(X_train, feature_subset, axis=1)
    X_valid = _safe_indexing(X_valid, feature_subset, axis=1)
    X_test = _safe_indexing(X_test, feature_subset, axis=1)
    model = model.fit(X_train, y_train)

    file_path = os.path.join(config.experiment_folder, f"{config.experiment_name}_feature_subset.joblib")
    joblib.dump(model, file_path)

    # Evaluate model
    result, y_preds, y_scores = sklearn_evaluate(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        f"{config.experiment_name}_feature_subset",
        features_subset,
    )
    results.append(result)

    # Save CSV
    columns = ["experiment", "train_acc", "train_auc", "valid_acc", "valid_auc", "test_acc", "test_auc", "features"]
    df = pd.DataFrame(results, columns=columns)
    file_path = os.path.join(config.experiment_folder, f"{config.experiment_name}_feature_selection.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved results to: {file_path}")

    columns = ["case_id", "subset", "y_pred", "y_score"]
    case_ids = dataset.case_ids()
    subsets = dataset.subsets()
    data = list(zip(case_ids, subsets, y_preds, y_scores))
    df = pd.DataFrame(data, columns=columns)
    file_path = os.path.join(config.experiment_folder, f"{config.experiment_name}-feature-selection-predictions.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved predictions to: {file_path}")


def sklearn_evaluate(model, X_train, y_train, X_valid, y_valid, X_test, y_test, experiment_name, features):
    results = [experiment_name]
    y_preds = []
    y_scores = []
    print(f"Evaluating: {experiment_name}")
    result, y_pred, y_score = sklearn_evaluate_subset(model, X_train, y_train, "Train")
    results += result
    y_preds += y_pred.tolist()
    y_scores += y_score.tolist()

    result, y_pred, y_score = sklearn_evaluate_subset(model, X_valid, y_valid, "Valid")
    results += result
    y_preds += y_pred.tolist()
    y_scores += y_score.tolist()

    result, y_pred, y_score = sklearn_evaluate_subset(model, X_test, y_test, "Test")
    results += result
    y_preds += y_pred.tolist()
    y_scores += y_score.tolist()

    results += [", ".join(features)]
    return results, y_preds, y_scores


def sklearn_evaluate_subset(model, X, y, label):
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    score = y_score.reshape(len(y_pred), 1)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, score)
    print(label)
    print(f"  Accuracy: {acc:.2f}")
    print(f"  ROC AUC: {auc:.2f}")
    print("")

    return [acc, auc], y_pred, y_score
