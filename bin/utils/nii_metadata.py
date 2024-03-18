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

import argparse
import json
import os.path

import nibabel as nib
import pandas as pd


def fetch_metadata(path: str, case_id: str, scan_type: str):
    img_path = os.path.join(path, f"{case_id}_{scan_type}.nii.gz")
    header = nib.load(img_path, mmap=False).header
    pixdim = list(header.get_zooms())
    return {"VoxelSize": [float(pixdim[0]), float(pixdim[1]), float(pixdim[2])]}


def main(csv_path: str, data_path: str):
    df = pd.read_csv(csv_path)
    results = []

    for i, row in df.iterrows():
        case_id = row["case_id"]
        ct_exists = row["ct_exists"]
        cta_exists = row["cta_exists"]
        centre = row["centre"]
        ct_path = os.path.normpath(os.path.join(data_path, centre, case_id, "ax_CT"))
        cta_path = os.path.normpath(os.path.join(data_path, centre, case_id, "ax_A"))

        print(f"Fetching stats for {case_id}")

        if ct_exists:
            ct_stats = fetch_metadata(ct_path, case_id, "ax_CT")
            ct_stats["ScanType"] = "ax_CT"
            ct_stats["Centre"] = centre
            results.append(ct_stats)

        if cta_exists:
            cta_stats = fetch_metadata(cta_path, case_id, "ax_A")
            cta_stats["ScanType"] = "ax_A"
            cta_stats["Centre"] = centre
            results.append(cta_stats)

    with open("nii-stats.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect scanner statistics from DICOM files")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument("data_path", type=str, help="The data directory that contains the DICOM files")

    args = parser.parse_args()

    main(args.csv_path, args.data_path)
