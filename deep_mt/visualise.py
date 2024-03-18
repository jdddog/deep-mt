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
import gc
import logging
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
import torch
from matplotlib.pyplot import savefig
from monai.data.dataset import Subset
from monai.visualize import matshow3d, blend_images
from torch.utils.data import DataLoader

from deep_mt.config import Config
from deep_mt.dataset import DeepMTDataset
from deep_mt.ml_utils import make_model, make_transforms
from deep_mt.monai_utils import set_determinism


def matsave3d(
    image: np.ndarray,
    path: str,
    title: str = None,
    dpi: int = 96,
    every_n: int = 5,
    cmap="gray",
    frame_dim: int = -3,
    channel_dim: int = None,
):
    # Transpose to axial view
    if len(image.shape) == 4:
        t = (0, 3, 2, 1)
    elif len(image.shape) == 5:
        t = (0, 1, 4, 3, 2)
    image = image.transpose(t)

    fig = plt.figure(facecolor="white")
    matshow3d(
        fig=fig,
        volume=image,
        title=title,
        every_n=every_n,
        show=False,
        cmap=cmap,
        frame_dim=frame_dim,
        channel_dim=channel_dim,
    )
    savefig(path, dpi=dpi, bbox_inches="tight", transparent=False, facecolor=fig.get_facecolor(), pad_inches=0.1)

    # Clear memory
    plt.cla()
    plt.clf()
    plt.close("all")
    gc.collect()


def matplot3d(
    image: np.ndarray,
    every_n: int = 5,
    cmap="gray",
    frame_dim: int = -3,
    channel_dim: int = None,
):
    # Transpose to axial view
    if len(image.shape) == 4:
        t = (0, 3, 2, 1)
    elif len(image.shape) == 5:
        t = (0, 1, 4, 3, 2)
    image = image.transpose(t)

    fig = plt.figure()
    matshow3d(
        fig=fig,
        volume=image,
        every_n=every_n,
        show=True,
        cmap=cmap,
        frame_dim=frame_dim,
        channel_dim=channel_dim,
    )

    # Clear memory
    plt.cla()
    plt.clf()
    plt.close("all")
    gc.collect()


def visualise(
    *,
    config: Config,
    subsets: List[str],
    dpi: int,
    every_n: int,
    n_cases: int = None,
    channel: int = None,
    predictions: pd.DataFrame = None,
):
    """Visualise the scans or transformed defined in a config file.

    :param config: the Config instance.
    :param type: the type of visualisation to create: scans or transforms.
    :param subsets: the dataset subsets to visualise, by default all subsets.
    :param dpi: the resolution in dots per inch.
    :param every_n: every n slices are displayed.
    :param n_cases: the limit on the number of cases to visualise.
    :param channel: a particular channel to visualise.
    :param spatial_size: the spatial size of each image when visualising scans.
    :return: None.
    """

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
    )

    # Print dataset summary
    dataset.print_train_summary()
    dataset.print_valid_summary()
    dataset.print_test_summary()

    # Define transforms
    train_transforms, valid_transforms = make_transforms(config)

    # Create visualisations
    root_folder = os.path.normpath(os.path.join(config.experiment_folder, "vis"))
    for subset in subsets:
        # Create datasets
        if subset == "train":
            ds = dataset.pytorch_train(train_transforms)
        elif subset == "valid":
            ds = dataset.pytorch_valid(valid_transforms)
        else:
            ds = dataset.pytorch_test(valid_transforms)

        # Make folder where outputs will be saved
        i = 0
        folder = os.path.join(root_folder, subset)
        os.makedirs(folder, exist_ok=True)

        # Load data
        loader = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.n_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for batch in loader:
            case_ids, images, labels = (
                batch["case_id"],
                batch[config.output_key],
                batch[config.target_class],
            )

            for case_id, image, label in zip(case_ids, images, labels):
                if n_cases is None or i < n_cases:
                    # Make title
                    class_names = dataset.class_names
                    y_true_label = class_names[label]
                    category = y_true_label
                    if predictions is not None:
                        y_pred = predictions[predictions["case_id"] == case_id].iloc[0]["class_id"]
                        y_pred_label = class_names[y_pred]
                        category += f" -> {y_pred_label}"
                    title = f"{case_id}: {category}"

                    # Select a particular channel to visualise
                    image = image.cpu().detach().numpy()
                    if channel is not None:
                        n_channels = image.shape[0]
                        assert channel < n_channels, f"channel={channel} must be < n_channels={n_channels}"
                        image = np.array([image[channel]])

                    if predictions is not None:
                        category_folder = os.path.join(folder, category)
                        os.makedirs(category_folder, exist_ok=True)
                        file_name = os.path.join(folder, category_folder, f"{case_id}.png")
                    else:
                        file_name = os.path.join(folder, f"{case_id}.png")

                    print(f"Visualising {case_id} and saving to {file_name}")
                    matsave3d(image, file_name, title=title, dpi=dpi, every_n=every_n)

                    i += 1
                else:
                    break

            if n_cases is not None and i >= n_cases:
                break

    # Print location of files
    print("")
    print(f"Visualisations saved to: {root_folder}")
    print("")


def visualise_salience(
    *,
    config: Config,
    weights_file: str,
    salience_type: str,
    subsets: List[str],
    case_ids: set[str],
    dpi: int,
    every_n: int,
    n_cases: int = None,
):
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
    print(f"Visualising salience for weights file: {weights_file}")

    #
    batch_size = config.batch_size
    if salience_type == "occlusion":
        batch_size = 1

    # Create visualisations
    class_names = dataset.class_names
    root_folder = os.path.normpath(os.path.join(config.experiment_folder, "salience", salience_type))
    for subset in subsets:
        # Create a test data loader
        if subset == "train":
            ds = dataset.pytorch_train(trans_valid)
        elif subset == "valid":
            ds = dataset.pytorch_valid(trans_valid)
        else:
            ds = dataset.pytorch_test(trans_valid)

        if case_ids is not None:
            ds = Subset(ds, indices=[i for i, item in enumerate(ds.data) if item["case_id"] in case_ids])

        data_loader = DataLoader(
            ds, batch_size=batch_size, num_workers=config.n_workers, pin_memory=torch.cuda.is_available()
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

        # Make salience visualisation instances
        target_layer = "class_layers.relu"
        grad_cam = monai.visualize.class_activation_maps.GradCAM(nn_module=model, target_layers=target_layer)
        grad_cam_pp = monai.visualize.class_activation_maps.GradCAMpp(nn_module=model, target_layers=target_layer)
        occ_sens = monai.visualize.OcclusionSensitivity(nn_module=model, n_batch=batch_size)

        # Make folder where outputs will be saved
        folder = os.path.join(root_folder, subset)
        os.makedirs(folder, exist_ok=True)

        # Load data
        i = 0
        for batch in data_loader:
            # Get data
            images, labels = batch[config.output_key].to(device), batch[config.target_class].to(device)

            # Predict outputs
            tab_data = None
            with torch.no_grad():
                if config.model_name == "deep_mt.hybrid_model.Densenet121TabularHybrid":
                    tab_data = batch["clinical_features"].to(device)
                    outputs = model(images, tab_data)
                else:
                    outputs = model(images)

            # Create target classes
            y_true = labels.cpu().detach().numpy()
            y_pred = np.argmax(outputs.cpu().detach().numpy(), axis=1)

            # Output salience
            if salience_type == "gradcam":
                salience_images = grad_cam(x=images, tab_data=tab_data, class_idx=labels)
            elif salience_type == "gradcam++":
                salience_images = grad_cam_pp(x=images, tab_data=tab_data, class_idx=labels)
            elif salience_type == "occlusion":
                salience_images, occ_most_prob = occ_sens(x=images, tab_data=tab_data)
                new_shape = (batch_size, 1) + images.shape[2:]
                salience_images = salience_images[0, y_true[0]][None].reshape(new_shape)
            else:
                raise ValueError(f"Salience type={salience_type} unknown")

            # For each image,
            case_ids = batch["case_id"]
            for case_id_, y_true_, y_pred_, img_, sal_ in zip(case_ids, y_true, y_pred, images, salience_images):
                if n_cases is None or i < n_cases:
                    # Get data
                    img_ = img_.cpu().detach().numpy()
                    sal_ = sal_.cpu().detach().numpy()

                    # Blend
                    image_blend = []
                    for c in img_:
                        img_channel = np.array([c])
                        channel_blend = blend_images(image=img_channel, label=sal_, alpha=0.5, rescale_arrays=True)
                        image_blend.append(channel_blend)
                    image_blend = np.array(image_blend)

                    # Make labels
                    y_true_label = class_names[y_true_]
                    y_pred_label = class_names[y_pred_]
                    category = f"{y_true_label} -> {y_pred_label}"
                    title = f"{case_id_}: {category}"

                    # Save
                    category_folder = os.path.join(folder, category)
                    os.makedirs(category_folder, exist_ok=True)
                    file_path = os.path.join(category_folder, f"{case_id_}.png")
                    print(f"Visualising salience {case_id_} and saving to {file_path}")

                    channel_dim = config.in_channels - 1
                    matsave3d(
                        image_blend,
                        file_path,
                        title=title,
                        dpi=dpi,
                        every_n=every_n,
                        cmap="hsv",
                        channel_dim=channel_dim,
                    )

                    i += 1
                else:
                    break

            if n_cases is not None and i >= n_cases:
                break

    # Print location of files
    print("")
    print(f"Visualisations saved to: {root_folder}")
    print("")
