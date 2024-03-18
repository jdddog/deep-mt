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

import os
from typing import List

from monai.data.image_reader import ITKReader

DICOM_EXTENSION = ".dcm"


def load_dicom(path: str):
    """Load a DICOM file from disk.

    :param path: the path to the folder containing the DICOM files.
    :return: the raw DICOM scan and headers.
    """

    reader = ITKReader()
    img = reader.read(path, fallback_only=False)
    scan_raw, header_raw = reader.get_data(img)
    return scan_raw, header_raw


def list_dicom_scan_paths(path: str) -> List[str]:
    """Given a path, lists the DICOM scans within the path, recursively.

    :param path: the path to scan.
    :return: a list of paths that contain DICOM scans.
    """

    paths = []

    for root, dirs, files in os.walk(path):
        for file in files:
            # If the file is a dicom file, then add the path and break
            if file.endswith(DICOM_EXTENSION):
                paths.append(root)
                break

    return paths


def folder_contains_dicoms(folder_path: str) -> bool:
    """Returns whether a folder contains DICOM files or not.

    :param folder_path: the path to the folder that might contain DICOM files.
    :return: whether the folder contains DICOM files.
    """

    contains_dicoms = False

    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(DICOM_EXTENSION):
                contains_dicoms = True
                break

    return contains_dicoms
