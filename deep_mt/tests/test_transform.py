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
from hashlib import sha1

import numpy as np
import torch
from monai.data import DataLoader
from monai.transforms import (
    AddChanneld,
    Compose,
    ConcatItemsd,
    DeleteItemsd,
    EnsureTyped,
    LoadImaged,
    MaskIntensityd,
    NormalizeIntensityd,
    RandAffined,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    SpatialCropd,
)

from deep_mt.dataset import DeepMTDataset
from deep_mt.monai_transforms import Clipd, MaskIntensity2d, RandClipd
from deep_mt.monai_utils import set_determinism
from deep_mt.someof import SomeOf
from deep_mt.transform import FillMissingImaged, make_transforms_mrs, make_transforms_mrs_v2, RandScaleIntensityRanged


def make_data_loader(transforms: Compose, seed: int, batch_size: int = 1, n_workers: int = 8):
    """Creates a data loader based on the validation dataset"""

    root_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))
    csv_path = os.path.join(root_path, "data/thrombectomy-2022-05-26-unique-patients.csv")
    scan_folder = os.path.join(root_path, "data/nii")
    dataset = DeepMTDataset(
        csv_path=csv_path,
        scan_folder=scan_folder,
        target_class="mrs02_36",
        ct_key="{case_id}_ax_CT_1.0x1.0x2.0mm_to_scct_unsmooth_SS_0_1.0x1.0x2.0mm_DenseRigid.nii.gz",
        cta_key="{case_id}_ax_A_cropped_to_{case_id}_ax_CT_1.0x1.0x2.0mm_to_scct_unsmooth_SS_0_1.0x1.0x2.0mm_DenseRigid.nii.gz",
        ct_mask_key="{case_id}_ax_CT_1.0x1.0x2.0mm_to_scct_unsmooth_SS_0_1.0x1.0x2.0mm_DenseRigid_combined_bet.nii.gz",
        train_ratio=0.9,
        valid_ratio=0.1,
        centre_test_set="ChCh",
        random_seed=seed,
    )
    dataset = dataset.pytorch_valid(transforms)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return data_loader


class TestDataset(unittest.TestCase):
    def test_make_transforms_mrs(self):
        input_shape = (32, 32, 32)
        output_shape = (32, 32, 32)

        # Check that correct transforms are composed
        # ct=True
        train, valid = make_transforms_mrs(input_shape=input_shape, output_shape=output_shape, ct=True)
        train_expected = [LoadImaged, AddChanneld, ScaleIntensityRanged, EnsureTyped]
        self.assertEqual(train_expected, [type(t) for t in train.transforms])
        self.assertEqual(train_expected, [type(t) for t in valid.transforms])

        # ct=True, cta=True
        train, valid = make_transforms_mrs(input_shape=input_shape, output_shape=output_shape, ct=True, cta=True)
        train_expected = [
            LoadImaged,
            FillMissingImaged,
            AddChanneld,
            ScaleIntensityRanged,
            ConcatItemsd,
            DeleteItemsd,
            EnsureTyped,
        ]
        self.assertEqual(train_expected, [type(t) for t in train.transforms])
        self.assertEqual(train_expected, [type(t) for t in valid.transforms])

        # ss=True
        train, valid = make_transforms_mrs(input_shape=input_shape, output_shape=output_shape, ct=True, ss=True)
        train_expected = [LoadImaged, AddChanneld, ScaleIntensityRanged, MaskIntensityd, DeleteItemsd, EnsureTyped]
        self.assertEqual(train_expected, [type(t) for t in train.transforms])
        self.assertEqual(train_expected, [type(t) for t in valid.transforms])

        # ventricles=True
        train, valid = make_transforms_mrs(input_shape=input_shape, output_shape=output_shape, ct=True, ventricles=True)
        train_expected = [LoadImaged, AddChanneld, ScaleIntensityRanged, SpatialCropd, EnsureTyped]
        self.assertEqual(train_expected, [type(t) for t in train.transforms])
        self.assertEqual(train_expected, [type(t) for t in valid.transforms])

        # rand=True
        train, valid = make_transforms_mrs(input_shape=input_shape, output_shape=output_shape, ct=True, rand=True)
        train_expected = [LoadImaged, AddChanneld, RandShiftIntensityd, RandAffined, ScaleIntensityRanged, EnsureTyped]
        valid_expected = [LoadImaged, AddChanneld, ScaleIntensityRanged, EnsureTyped]
        self.assertEqual(train_expected, [type(t) for t in train.transforms])
        self.assertEqual(valid_expected, [type(t) for t in valid.transforms])

        # rand=True settings
        # Train: RandShiftIntensityd
        expected_offset = (-2, 2)
        shifter = train.transforms[2].shifter
        self.assertEqual(expected_offset, shifter.offsets)

        # Train: RandAffined
        rand_trans_x = 21
        rand_trans_y = 25
        rand_trans_z = 5
        rand_rotate_offset = 2 * np.pi / 180
        rand_scale_ratio = 0.05
        rand_affine = train.transforms[3].rand_affine
        rand_affine_grid = rand_affine.rand_affine_grid
        self.assertEqual(rand_affine_grid.rotate_range, (rand_rotate_offset, rand_rotate_offset, rand_rotate_offset))
        self.assertEqual(
            rand_affine_grid.translate_range,
            (rand_trans_x, rand_trans_y, rand_trans_z),
        )
        self.assertEqual(rand_affine_grid.scale_range, (rand_scale_ratio, rand_scale_ratio, rand_scale_ratio))

        # Train: ScaleIntensityRanged
        scaler = train.transforms[4].scaler
        self.assertEqual(-1000, scaler.a_min)
        self.assertEqual(1000, scaler.a_max)
        self.assertEqual(0, scaler.b_min)
        self.assertEqual(1, scaler.b_max)
        self.assertTrue(scaler.clip)

        # Valid: ScaleIntensityRanged
        scaler = valid.transforms[2].scaler
        self.assertEqual(-1000, scaler.a_min)
        self.assertEqual(1000, scaler.a_max)
        self.assertEqual(0, scaler.b_min)
        self.assertEqual(1, scaler.b_max)
        self.assertTrue(scaler.clip)

        # dynamic_window=True
        train, valid = make_transforms_mrs(
            input_shape=input_shape, output_shape=output_shape, ct=True, dynamic_window=True
        )
        train_expected = [LoadImaged, AddChanneld, RandScaleIntensityRanged, EnsureTyped]
        valid_expected = [LoadImaged, AddChanneld, ScaleIntensityRanged, EnsureTyped]
        self.assertEqual(train_expected, [type(t) for t in train.transforms])
        self.assertEqual(valid_expected, [type(t) for t in valid.transforms])

        # resize=True
        train, valid = make_transforms_mrs(input_shape=input_shape, output_shape=output_shape, ct=True, resize=True)
        train_expected = [LoadImaged, AddChanneld, ScaleIntensityRanged, Resized, EnsureTyped]
        self.assertEqual(train_expected, [type(t) for t in train.transforms])
        self.assertEqual(train_expected, [type(t) for t in valid.transforms])

    def test_make_transforms_mrs_v2(self):
        input_shape = (32, 32, 32)
        output_shape = (32, 32, 32)

        # Check that correct transforms are composed
        # ct=True
        train, valid = make_transforms_mrs_v2(input_shape=input_shape, output_shape=output_shape, ct=True)
        train_expected = [LoadImaged, AddChanneld, RandClipd, SomeOf, NormalizeIntensityd, EnsureTyped]
        valid_expected = [LoadImaged, AddChanneld, Clipd, NormalizeIntensityd, EnsureTyped]
        self.assertEqual(train_expected, [type(t) for t in train.transforms])
        self.assertEqual(valid_expected, [type(t) for t in valid.transforms])

        # ct=True, cta=True
        train, valid = make_transforms_mrs_v2(input_shape=input_shape, output_shape=output_shape, ct=True, cta=True)
        train_expected = [
            LoadImaged,
            FillMissingImaged,
            AddChanneld,
            ConcatItemsd,
            DeleteItemsd,
            RandClipd,
            SomeOf,
            NormalizeIntensityd,
            EnsureTyped,
        ]
        valid_expected = [
            LoadImaged,
            FillMissingImaged,
            AddChanneld,
            ConcatItemsd,
            DeleteItemsd,
            Clipd,
            NormalizeIntensityd,
            EnsureTyped,
        ]
        self.assertEqual(train_expected, [type(t) for t in train.transforms])
        self.assertEqual(valid_expected, [type(t) for t in valid.transforms])

        # ss=True
        train, valid = make_transforms_mrs_v2(input_shape=input_shape, output_shape=output_shape, ss=True)
        train_expected = [
            LoadImaged,
            AddChanneld,
            RandClipd,
            SomeOf,
            MaskIntensity2d,
            DeleteItemsd,
            NormalizeIntensityd,
            EnsureTyped,
        ]
        valid_expected = [
            LoadImaged,
            AddChanneld,
            Clipd,
            MaskIntensity2d,
            DeleteItemsd,
            NormalizeIntensityd,
            EnsureTyped,
        ]
        self.assertEqual(train_expected, [type(t) for t in train.transforms])
        self.assertEqual(valid_expected, [type(t) for t in valid.transforms])

        # ventricles=True
        train, valid = make_transforms_mrs_v2(input_shape=input_shape, output_shape=output_shape, ventricles=True)
        train_expected = [
            LoadImaged,
            AddChanneld,
            RandClipd,
            SomeOf,
            SpatialCropd,
            NormalizeIntensityd,
            EnsureTyped,
        ]
        valid_expected = [
            LoadImaged,
            AddChanneld,
            Clipd,
            SpatialCropd,
            NormalizeIntensityd,
            EnsureTyped,
        ]
        self.assertEqual(train_expected, [type(t) for t in train.transforms])
        self.assertEqual(valid_expected, [type(t) for t in valid.transforms])

    @unittest.skipIf("GITHUB_ACTIONS" in os.environ, "Test should only be run locally")
    def test_transforms_deterministic(self):
        seed = 7
        input_shape = (181, 217, 90)
        output_shape = (128, 160, 40)

        runs = []
        for _ in range(2):
            set_determinism(seed)
            examples = []

            # Test pipeline with the train dataset as this has the random transforms
            train, _ = make_transforms_mrs(
                input_shape=input_shape, output_shape=output_shape, ct=True, rand=True, dynamic_window=True
            )
            data_loader = make_data_loader(train, seed)

            # Load data
            for batch in data_loader:
                scans = batch["ct_image"]
                for scan in scans:
                    content_id = sha1(np.ascontiguousarray(scan.numpy())).hexdigest()
                    examples.append(content_id)
            runs.append(examples)

        self.assertEqual(runs[0], runs[1])
