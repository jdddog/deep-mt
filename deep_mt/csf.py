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
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import ray
from monai.transforms import LoadImage

from deep_mt.data_cleaning import yield_tasks


def make_csf_brain_mask(img: np.ndarray, brain_mask: np.ndarray, csf_min: int, csf_max: int) -> np.ndarray:
    """Make CSF mask.

    :param img: the CT scan in Hounsfield units.
    :param brain_mask: the mask of the brain.
    :param csf_min: the minimum value to consider as CSF.
    :param csf_max: the maximum value to consider as CSF.
    :return: the mask as a numpy array.
    """

    return np.array((img >= csf_min) & (img <= csf_max) & brain_mask.astype(bool), dtype=np.int8)


def calc_volume(mask: np.ndarray, voxel_volume: float):
    """Calculate the volume of a mask.

    :param mask: the mask.
    :param voxel_volume: the volume of a single voxel.
    :return: the voxel volume.
    """

    return (np.count_nonzero(mask) * voxel_volume) / 1000000.0


def calc_case_csf(
    img: np.ndarray, brain_mask: np.ndarray, meta: Dict, csf_min: int, csf_max: int
) -> Tuple[float, float, float]:
    """

    :param img:
    :param brain_mask:
    :param meta:
    :param csf_min:
    :param csf_max:
    :return:
    """

    # Voxel volume
    pixdim = meta["pixdim"]
    x, y, z = pixdim[1], pixdim[2], pixdim[3]
    voxel_volume = x * y * z

    # Calculate stats
    brain_volume = calc_volume(brain_mask, voxel_volume)
    csf_masks = make_csf_brain_mask(img, brain_mask, csf_min, csf_max)
    csf_volume = calc_volume(csf_masks, voxel_volume)
    csf_ratio = csf_volume / brain_volume

    return brain_volume, csf_volume, csf_ratio


@ray.remote
def calc_case_csf_task(case_id: str, img_path: str, mask_path: str, csf_min: int, csf_max: int):
    brain_volume, csf_volume, csf_ratio = None, None, None
    if os.path.isfile(img_path) and os.path.isfile(mask_path):
        img, img_meta = LoadImage()(img_path)
        mask, mask_meta = LoadImage()(mask_path)
        brain_volume, csf_volume, csf_ratio = calc_case_csf(img, mask, img_meta, csf_min, csf_max)
    return case_id, brain_volume, csf_volume, csf_ratio


def calc_csf(csv_file, nii_path: str, output_path: str, csf_min: int, csf_max: int):
    """

    :param csv_file:
    :param nii_path:
    :param output_path:
    :param csf_min:
    :param csf_max:
    :return:
    """

    df = pd.read_csv(csv_file)
    scan_type = "ax_CT"

    # Create jobs
    task_ids = []
    index = {}
    for i, row in df.iterrows():
        centre = row["centre"]
        case_id = row["case_id"]
        print(f"Processing case: {case_id}")
        case_path = os.path.join(nii_path, centre, case_id, scan_type)
        img_path = os.path.join(case_path, f"{case_id}_{scan_type}.nii.gz")
        mask_path = os.path.join(case_path, f"{case_id}_{scan_type}_combined_bet.nii.gz")
        task_id = calc_case_csf_task.remote(case_id, img_path, mask_path, csf_min, csf_max)
        task_ids.append(task_id)

    # Wait for jobs
    print("Waiting for results:")
    for task in yield_tasks(task_ids):
        case_id = task[0]
        print(f"Processed: {case_id}")
        index[case_id] = task

    # Collect results in same order as CSV
    data = []
    for i, row in df.iterrows():
        case_id = row["case_id"]
        data.append(index[case_id])

    df_out = pd.DataFrame(data, columns=["case_id", "brain_volume", "csf_volume", "csf_ratio"])
    df_out.to_csv(output_path, index=False)
