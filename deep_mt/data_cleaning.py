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

import hashlib
import logging
import os
import shutil
from typing import Dict, Tuple, Optional, List

import SimpleITK as sitk
import click
import numpy as np
import pandas as pd
import ray
from natsort.natsort import natsorted

from deep_mt.dicom import folder_contains_dicoms
from deep_mt.dicom import list_dicom_scan_paths, load_dicom

DICOM_EXTENSION = ".dcm"


def csv_file_summary(csv_file: str):
    """Print a summary of the CSV file.

    :param csv_file: the CSV file path.
    :return: None.
    """

    df = pd.read_csv(csv_file)
    for column_name in df.columns:
        total = len(df[column_name])
        print(f"{column_name}: {total}")
        for class_name, class_count in df[column_name].value_counts().items():
            print(f"  {class_name}: {class_count}")
        print("")


def merge(src: str, dst: str):
    """Merge two directories of DICOM scans.

    :param src: the source directory.
    :param dst: the destination directory.
    :return: None.
    """

    paths = list_dicom_scan_paths(src)
    suffixes = ["ax_CT", "ax_A"]

    # List changes
    merges = []
    for src_path in paths:
        include_path = any([src_path.endswith(suffix) for suffix in suffixes])
        if include_path:
            dst_path = os.path.join(dst, src_path[len(src) + 1 :])
            dst_path_remove = os.path.exists(dst_path)
            merges.append((src_path, dst_path_remove, dst_path))

    if not len(merges):
        print("No merge actions found")
        exit(0)

    # Print merges
    print("Merge actions:")
    for src_path, dst_path_remove, dst_path in merges:
        print(f"\tsrc_path: {src_path}")
        print(f"\tdst_path_remove: {dst_path_remove}")
        print(f"\tdst_path: {dst_path}")
        print("")

    # Confirm
    if click.confirm("Do you want to execute the merge actions?"):
        print("Merging folders:")
        for src_path, dst_path_remove, dst_path in merges:
            # Remove destination if it exists
            if dst_path_remove:
                shutil.rmtree(dst_path)
                print(f"\tremoved: {dst_path}")

            # Move source to destination
            shutil.move(src_path, dst_path)
            print(f"\tmoved: {src_path} to {dst_path}")
            print("")


def check_dicom_readable(scan_path) -> Tuple[bool, Optional[Exception]]:
    """

    :param scan_path:
    :return:
    """

    try:
        load_dicom(scan_path)
        return True, None
    except Exception as e:
        return False, e


def get_dicom_series_ids(dir_name: str) -> List[str]:
    return sitk.ImageSeriesReader_GetGDCMSeriesIDs(dir_name)


@ray.remote
def check_scan_readable(scan_path: str):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"check_scan_readable: {scan_path}")

    try:
        # Check if multiple exist
        series_ids = get_dicom_series_ids(scan_path)
        print(series_ids)
        if len(series_ids) == 1:
            # Check if readable
            is_readable, e = check_dicom_readable(scan_path)
            if is_readable:
                return scan_path, True, None
            else:
                return scan_path, False, e
        elif len(series_ids) == 0:
            msg = f"No series detected in scan folder {scan_path}"
            return scan_path, False, msg
        else:
            msg = f"Multiple series detected in scan folder {scan_path}: {series_ids}"
            print(msg)
            return scan_path, False, msg
    except Exception as e:
        logging.error(f"Error processing scan: {scan_path}, {e}")

    return scan_path, False, e


def check_readable(path: str):
    """Check that all DICOMs are readable.

    :param path: the path.
    :return: None.
    """

    # List scans
    scan_paths = list_dicom_scan_paths(path)
    scan_paths = natsorted(scan_paths)

    # Create slice index
    print("Checking scans:")
    task_ids = []
    for scan_path in scan_paths:
        print(f"Processing: {scan_path}")
        task_id = check_scan_readable.remote(scan_path)
        task_ids.append(task_id)

    # Process results
    readable = []
    not_readable = []
    for scan_path, is_readable, e in yield_tasks(task_ids):
        if is_readable:
            readable.append(scan_path)
        else:
            not_readable.append((scan_path, e))

    print("Readable scans:")
    for scan_path in readable:
        print(f"\t{scan_path}")

    print("Unreadable scans:")
    for scan_path, e in not_readable:
        print(f"\t{scan_path}")
        print(f"\t{scan_path}")

    print("Reasons unreadable:")
    for scan_path, e in not_readable:
        print(f"\t{scan_path}")
        print(f"\t{e}")


@ray.remote
def load_scan_task(scan_path: str):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Processing scan: {scan_path}")
    slice_index = dict()

    try:
        scan_raw, header_raw = load_dicom(scan_path)
        scan_raw = np.ascontiguousarray(scan_raw)
        for slice_ in scan_raw:
            is_uniform = np.all(slice_ == slice_[0][0])
            if not is_uniform:
                id_ = slice_id(slice_)
                if id_ in slice_index:
                    slice_index[id_].add(scan_path)
                else:
                    slice_index[id_] = {scan_path}

        logging.info(f"Finished scan: {scan_path}")
    except Exception as e:
        logging.error(f"Error processing scan: {scan_path}, {e}")

    return slice_index


def find_duplicates(path: str) -> Dict:
    """Find duplicate DICOM scans and print them to the screen.

    :return: a dictionary containing the list of duplicate DICOM scans.
    """

    # Setup logging and ray
    logging.basicConfig(level=logging.INFO)
    # ray.init(num_cpus=1, local_mode=True)

    # List scans
    scan_paths = list_dicom_scan_paths(path)

    # Create slice index
    print("Hashing scans:")
    task_ids = []
    for scan_path in scan_paths:
        print(f"  {scan_path}")
        task_id = load_scan_task.remote(scan_path)
        task_ids.append(task_id)

    # Merge slice indexes
    slice_index = dict()
    for slice_index_subset in yield_tasks(task_ids):
        for key, val in slice_index_subset.items():
            if key in slice_index:
                slice_index[key] |= slice_index_subset[key]
            else:
                slice_index[key] = slice_index_subset[key]

    # Find duplicate scans
    print("\nIndexing duplicate scans...")
    duplicate_index = dict()
    for key, scan_ids in slice_index.items():
        if len(scan_ids) > 1:
            dup_id = tuple(scan_ids)
            if dup_id not in duplicate_index:
                duplicate_index[dup_id] = True

    # Print duplicates
    print("\nDuplicate scans:")
    for scan_ids in duplicate_index.keys():
        print("  Duplicate set:")
        for scan_id in scan_ids:
            print(f"    {scan_id}")
        print("")

    return duplicate_index


def find_missing_dicoms(csv_file, dicoms_path: str, output_path: str):
    df = pd.read_csv(csv_file)
    data = []
    for i, row in df.iterrows():
        centre = row["centre"]
        case_id = row["case_id"]
        case_path = os.path.join(dicoms_path, centre, case_id)
        ct_exists = folder_contains_dicoms(os.path.join(case_path, "ax_CT"))
        cta_exists = folder_contains_dicoms(os.path.join(case_path, "ax_A"))
        data.append((case_id, int(ct_exists), int(cta_exists)))

    df_out = pd.DataFrame(data, columns=["case_id", "ct_exists", "cta_exists"])
    df_out.to_csv(output_path, index=False)


def yield_tasks(task_ids, timeout=10.0):
    """Yield ray tasks.

    :param task_ids: the task ids.
    :param timeout: timeout for waiting.
    :return: None.
    """

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    results_count = 0
    while True:
        ready_ids, not_ready_ids = ray.wait(task_ids, num_returns=len(task_ids), timeout=timeout)

        # Add the results that have completed
        for ready_id in ready_ids:
            result = ray.get(ready_id)
            results_count += 1
            yield result
        task_ids = not_ready_ids

        logging.info(f"Num tasks complete: {results_count}, num tasks waiting: {len(task_ids)}.")

        # If no tasks left then break
        if len(task_ids) == 0:
            break


def slice_id(slice: np.ndarray):
    """Make a unique id for a given DICOM slice.

    :param slice: the DICOM slice.
    :return: the unique id.
    """

    hash_ = hashlib.md5()
    hash_.update(slice)
    return hash_.hexdigest()
