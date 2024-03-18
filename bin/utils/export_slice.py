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

import SimpleITK as sitk


def extract_slice_to_png(input_file, output_file, slice_index, min_hu, max_hu):
    # Read the .nii image
    image = sitk.ReadImage(input_file)

    # Extract a 2D slice (Axial view used as an example)
    slice_image = image[:, :, slice_index]

    # Rotate the slice by 180 degrees
    # slice_image = sitk.Flip(slice_image, [True, True])

    # Apply window/level
    rescaler = sitk.IntensityWindowingImageFilter()
    rescaler.SetOutputMaximum(255)
    rescaler.SetOutputMinimum(0)
    rescaler.SetWindowMaximum(max_hu)  # Window
    rescaler.SetWindowMinimum(min_hu)  # Level
    slice_windowed = rescaler.Execute(slice_image)

    slice_windowed = sitk.Cast(slice_windowed, sitk.sitkUInt8)

    # Save the slice to a PNG
    sitk.WriteImage(slice_windowed, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a specific slice from an NIfTI file and save as PNG with specified window/level."
    )

    parser.add_argument("input_file", type=str, help="Path to the input .nii file.")
    parser.add_argument("output_file", type=str, help="Path to the output PNG file.")
    parser.add_argument("slice_index", type=int, help="Index of the slice to extract.")
    parser.add_argument("min_hu", type=float, help="Min Hu for windowing")
    parser.add_argument("max_hu", type=float, help="Max Hu for windowing")

    args = parser.parse_args()

    extract_slice_to_png(args.input_file, args.output_file, args.slice_index, args.min_hu, args.max_hu)
