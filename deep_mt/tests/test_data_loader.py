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

import os
import unittest

import monai
import torch
from monai.data import DataLoader

from deep_mt.dataset import DeepMTDataset
from deep_mt.monai_utils import set_determinism
from deep_mt.transform import make_transforms_mrs_v2


def run_epochs(dataset, n_epochs: int = 2, **kwargs):
    # create a DataLoader
    dataloader = monai.data.DataLoader(dataset, **kwargs)

    # run through the dataloader twice and store the examples seen in each epoch
    examples_seen = []
    for _ in range(n_epochs):
        examples = []
        for batch in dataloader:
            examples.extend(batch)
        examples_seen.append(examples)

    return examples_seen


def run_epochs_real(seed: int, batch_size: int = 10, n_epochs: int = 2, n_workers: int = 8):
    root_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))
    csv_path = os.path.join(root_path, "data/thrombectomy-2022-05-26-unique-patients.csv")
    scan_folder = os.path.join(root_path, "data/nii")
    dataset = DeepMTDataset(
        csv_path=csv_path,
        scan_folder=scan_folder,
        target_class="mrs02_36",
        ct_key="{case_id}_ax_CT_1.0x1.0x2.0mm_to_scct_unsmooth_SS_0_1.0x1.0x2.0mm_DenseRigid.nii.gz",
        cta_key=None,
        # "{case_id}_ax_A_cropped_to_{case_id}_ax_CT_1.0x1.0x2.0mm_to_scct_unsmooth_SS_0_1.0x1.0x2.0mm_DenseRigid.nii.gz"
        ct_mask_key="{case_id}_ax_CT_1.0x1.0x2.0mm_to_scct_unsmooth_SS_0_1.0x1.0x2.0mm_DenseRigid_combined_bet.nii.gz",
        train_ratio=0.9,
        valid_ratio=0.1,
        centre_test_set="ChCh",
        random_seed=seed,
    )
    transforms, _ = make_transforms_mrs_v2(input_shape=(181, 217, 90), output_shape=(128, 160, 40), ct=True, cta=False)
    dataset = dataset.pytorch_train(transforms)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # run through the dataloader twice and store the examples seen in each epoch
    examples_seen = []
    for _ in range(n_epochs):
        examples = []
        for batch in data_loader:
            case_ids = batch["case_id"]
            examples.extend(case_ids)
        examples_seen.append(examples)

    return examples_seen


class TestDataLoader(unittest.TestCase):
    def test_deterministic_loading(self):
        # Create a dummy dataset with 1000 examples
        seed = 7
        dataset = monai.data.ArrayDataset(range(10))

        # Check if the examples seen in each epoch are different due to shuffling
        examples_seen = run_epochs(dataset, batch_size=5, shuffle=True)
        self.assertNotEqual(examples_seen[0], examples_seen[1])

        # Check if the examples seen in each epoch are the same due to no shuffling
        examples_seen = run_epochs(dataset, batch_size=5, shuffle=False)
        self.assertEqual(examples_seen[0], examples_seen[1])

        # # Check if the examples seen in each epoch are different in two different runs
        # examples_seen_1 = run_epochs(dataset, batch_size=5, shuffle=True)
        # examples_seen_2 = run_epochs(dataset, batch_size=5, shuffle=True)
        # self.assertNotEqual(examples_seen_1, examples_seen_2)

        # Check if the examples seen are the same even with shuffling due to set_determinism
        set_determinism(seed)
        examples_seen_1 = run_epochs(dataset, batch_size=5, shuffle=True)
        set_determinism(seed)
        examples_seen_2 = run_epochs(dataset, batch_size=5, shuffle=True)
        self.assertEqual(examples_seen_1, examples_seen_2)

        # Multiple workers
        num_workers = 10
        set_determinism(seed)
        examples_seen_1 = run_epochs(dataset, batch_size=5, shuffle=True, num_workers=num_workers)
        set_determinism(seed)
        examples_seen_2 = run_epochs(dataset, batch_size=5, shuffle=True, num_workers=num_workers)
        self.assertEqual(examples_seen_1, examples_seen_2)

    @unittest.skipIf("GITHUB_ACTIONS" in os.environ, "Test should only be run locally")
    def test_deterministic_loading_real_dataset(self):
        # Test that loading is deterministic with the real dataset and multiple workers
        seed = 7
        batch_size = 10
        n_epochs = 3
        n_workers = 12

        set_determinism(seed)
        examples_seen_1 = run_epochs_real(seed, n_workers=n_workers, batch_size=batch_size, n_epochs=n_epochs)

        set_determinism(seed)
        examples_seen_2 = run_epochs_real(seed, n_workers=n_workers, batch_size=batch_size, n_epochs=n_epochs)
        self.assertEqual(examples_seen_1, examples_seen_2)
