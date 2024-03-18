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
from functools import partial

import monai
import numpy as np
import torch
from lime.lime_tabular import LimeTabularExplainer
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from deep_mt.config import Config
from deep_mt.dataset import DeepMTDataset
from deep_mt.ml_utils import make_model, make_transforms
from deep_mt.monai_utils import set_determinism

# Load config
config_path = os.path.abspath(
    "../data/configs/combined/mrs-ct-cta-1.0x1.0x2.0mm-152x182x76px-fine-tune-squeeze-no-basilars-epoch-21.yaml"
)
weights_file = os.path.abspath(
    "../data/experiments/mrs-ct-cta-1.0x1.0x2.0mm-152x182x76px-fine-tune-squeeze-no-basilars-epoch-21/mrs-ct-cta-1.0x1.0x2.0mm-152x182x76px-fine-tune-squeeze-no-basilars-epoch-21_epoch_47.pth"
)
config = Config.load(config_path)

# Load dataset
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
    pre_process_std_scale=False,  # Turn std scale off, as LIME needs it to be off, however the model needs it to be on
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

# Make MONAI datasets
ds_train = dataset.pytorch_train(trans_valid)
ds_test = dataset.pytorch_test(trans_valid)

# Explainer
# age: continuous, std scaled
# nihss_baseline: ordinal, std scaled
# mrs_baseline: ordinal, std scaled
# bsl: continuous, std scaled
# onset_to_groin: continuous, std scaled
training_data = np.array([item["clinical_features"] for item in ds_train.data]).reshape(
    (len(ds_train), dataset.n_clinical_features)
)

explainer = LimeTabularExplainer(
    training_data=training_data, feature_names=dataset.clinical_feature_names, class_names=dataset.class_names
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_kwargs = copy.copy(config.model_kwargs)
if config.model_name == "deep_mt.hybrid_model.Densenet121TabularHybrid":
    model_kwargs["n_tab_features"] = dataset.n_clinical_features
    print(f"model_kwargs: {model_kwargs}")
    print(f"n_clinical_features: {dataset.n_clinical_features}")
    print(f"clinical_features: {dataset.clinical_feature_names}")

model = make_model(config.model_name, model_kwargs)
model.load_state_dict(torch.load(weights_file))
model = model.to(device)
model.eval()

# Create and fit std scaler on training data which will be used when running our model
scaler = StandardScaler()
scaler.fit(training_data)


def model_predict(image, data, batch_size: int = 64):
    # image is shape: 2, 152, 182, 76 -> B, 2, 152, 182, 76
    # data is shape: 5000, 5 -> B, 1, 5
    n_features = len(config.features)
    num_batches = int(np.ceil(len(data) / batch_size))

    with torch.no_grad():
        # Iterate over batches, creating predictions
        y_pred = []
        for i in tqdm(range(num_batches)):
            batch = data[i * batch_size : (i + 1) * batch_size]
            batch = scaler.transform(batch)
            actual_batch_size = len(batch)
            image_batch = image.unsqueeze(0).repeat(actual_batch_size, 1, 1, 1, 1).to(device)
            tab_batch = torch.tensor(batch.reshape(actual_batch_size, 1, n_features), dtype=torch.float32).to(device)
            y_pred_batch = model(image_batch, tab_batch).cpu().detach().numpy()
            y_pred.append(y_pred_batch)
            del image_batch, tab_batch

    y_pred = np.concatenate(y_pred)
    y_softmax = softmax(y_pred, axis=1)
    return y_softmax


def explain_case(case_id: str):
    item_index = dataset.index_from_case_id("test", case_id)
    item = ds_test[item_index]
    image = item[config.output_key]
    tab_data = item["clinical_features"][0]
    actual_case_id = item["case_id"]
    assert case_id == actual_case_id, "case ids do not match"
    predict_fn = partial(model_predict, image)
    exp = explainer.explain_instance(data_row=tab_data, predict_fn=predict_fn, num_samples=5000)
    exp.save_to_file(f"{case_id}.html", show_table=True)


if __name__ == "__main__":
    # Explain cases
    for case_id in ["my-case-id-1", "my-case-id-2"]:
        explain_case(case_id)
