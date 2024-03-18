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

import pandas as pd


def main(json_path: str):
    results = []
    with open(json_path) as f:
        data = json.load(f)

        for row in data:
            manufacturer = row["Manufacturer"].title().replace("Ge", "GE")
            manufacturer_model_name = f'{manufacturer} {row["ManufacturerModelName"]}'
            conv_kernel = row["ConvolutionKernel"]
            if isinstance(conv_kernel, list):
                conv_kernel = ",".join(conv_kernel)
            conv_kernel = f"{manufacturer} {conv_kernel}"
            filter_type = f'{manufacturer} {row["FilterType"]}'
            results.append(
                [row["Centre"], row["ScanType"], manufacturer, manufacturer_model_name, conv_kernel, filter_type]
            )

    df = pd.DataFrame(
        results,
        columns=["Centre", "ScanType", "Manufacturer", "ManufacturerModelName", "ConvolutionKernel", "FilterType"],
    )
    df = df[(df["ScanType"] == "ax_CT")]
    columns = ["Manufacturer", "ManufacturerModelName", "ConvolutionKernel"]
    for col in columns:
        df_group = df[col].value_counts().reset_index()
        df_group.columns = [col, "Count"]
        df_group.to_csv(f"{col}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect scanner statistics from DICOM files")
    parser.add_argument("json_path", type=str, help="Path to the input CSV file")
    args = parser.parse_args()
    main(args.json_path)
