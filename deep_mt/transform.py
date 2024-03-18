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
import math
from typing import List, Tuple

import numpy as np
from monai.transforms import (
    AddChanneld,
    Compose,
    ConcatItemsd,
    DeleteItemsd,
    EnsureTyped,
    LoadImaged,
    MaskIntensityd,
    NormalizeIntensityd,
    OneOf,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandStdShiftIntensityd,
    RandZoomd,
    Resized,
    ScaleIntensityRanged,
    SpatialCropd,
)

from deep_mt.monai_transforms import Clipd, FillMissingImaged, MaskIntensity2d, RandClipd, RandScaleIntensityRanged
from deep_mt.someof import SomeOf


def make_transforms_legacy(
    *,
    key: str,
    output_shape: Tuple[int, int, int],
    ct_key: str = "ct_image",
    cta_key: str = "cta_image",
    output_key: str = "ct_cta_image",
    ct_mask_key: str = "ct_mask",
    **kwargs,
):
    train, valid = [], []
    output_keys = [output_key]

    if key == "ct":
        # The following values gave good results in the previous experiments:
        a_min, a_max = -1000, 1000
        b_min, b_max = 0.0, 1.0
        input_keys = [ct_key]
        train = Compose(
            [
                LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True),
                AddChanneld(keys=input_keys),
                ScaleIntensityRanged(keys=input_keys, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
                Resized(keys=output_keys, spatial_size=output_shape),
                EnsureTyped(keys=output_keys),
            ]
        )
        valid = copy.deepcopy(train)
    elif key == "ct-ss":
        # The following values gave good results in the previous experiments:
        a_min, a_max = -1000, 1000
        b_min, b_max = 0.0, 1.0
        input_keys = [ct_key, ct_mask_key]
        train = Compose(
            [
                LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True),
                AddChanneld(keys=input_keys),
                ScaleIntensityRanged(keys=[ct_key], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
                MaskIntensityd(keys=[ct_key], mask_key=ct_mask_key),
                DeleteItemsd(keys=[ct_mask_key]),
                Resized(keys=output_keys, spatial_size=output_shape),
                EnsureTyped(keys=output_keys),
            ]
        )
        valid = copy.deepcopy(train)
    elif key == "ct-ss-rand-trans":
        # The following values gave good results in the previous experiments:
        a_min, a_max = -1000, 1000
        b_min, b_max = 0.0, 1.0
        input_keys = [ct_key, ct_mask_key]
        prob = 0.1
        train = Compose(
            [
                LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True),
                AddChanneld(keys=input_keys),
                RandShiftIntensityd(keys=[ct_key], offsets=5, prob=prob),
                ScaleIntensityRanged(keys=[ct_key], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
                MaskIntensityd(keys=[ct_key], mask_key=ct_mask_key),
                DeleteItemsd(keys=[ct_mask_key]),
                RandRotated(keys=[ct_key], range_x=math.pi / 180, prob=prob),
                RandZoomd(keys=[ct_key], min_zoom=0.975, max_zoom=1.025, prob=prob),
                Resized(keys=output_keys, spatial_size=output_shape),
                EnsureTyped(keys=output_keys),
            ]
        )
        valid = Compose(
            [
                LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True),
                AddChanneld(keys=input_keys),
                ScaleIntensityRanged(keys=[ct_key], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
                MaskIntensityd(keys=[ct_key], mask_key=ct_mask_key),
                DeleteItemsd(keys=[ct_mask_key]),
                Resized(keys=output_keys, spatial_size=output_shape),
                EnsureTyped(keys=output_keys),
            ]
        )
    elif key == "ct-cta":
        # The following values gave good results in the previous experiments:
        a_min, a_max = -1000, 1000
        b_min, b_max = 0.0, 1.0
        input_keys = [ct_key, cta_key]
        train = Compose(
            [
                LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True),
                # Fill the missing image with -1000 HU to prevent KeyError: 'ct_image_transforms'
                FillMissingImaged(keys=input_keys, constant=a_min, shape_key=ct_key),
                AddChanneld(keys=input_keys),
                ScaleIntensityRanged(keys=input_keys, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
                ConcatItemsd(keys=input_keys, name=output_key),
                DeleteItemsd(keys=input_keys),
                Resized(keys=output_keys, spatial_size=output_shape),
                EnsureTyped(keys=output_keys),
            ]
        )
        valid = copy.deepcopy(train)
    elif key == "ct-ventricles":
        # http://www.ieomsociety.org/singapore2021/papers/41.pdf
        # Suggests a window of -12.5 to 82.5 would be good
        a_offset = 5
        a_min, a_max = 13, 83
        a_min_range, a_max_range = (a_min - a_offset, a_min + a_offset), (a_max - a_offset, a_max + a_offset)
        b_min, b_max = 0.0, 1.0
        input_keys = [ct_key]

        # Crop to around the ventricles
        # Crop may be able to be a tighter around the skull
        # 291, 365, 48
        roi_start = [60, 70, 57]
        roi_end = [351, 435, 105]

        # Augmentation settings
        prob = 1.0
        intensity_offset = 2  # +- 2 Hu
        translate_range = (411 * 0.05, 493 * 0.05, 181 * 0.025)  # +-5%, +-5%, +-2.5% in each dimension X, Y, Z
        rotate_offset = 2 * np.pi / 180  # +- 2 degrees
        scale_ratio = 0.05  # +-5%

        train = Compose(
            [
                LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True),
                AddChanneld(keys=input_keys),
                RandShiftIntensityd(keys=input_keys, prob=prob, offsets=intensity_offset),
                RandAffined(
                    keys=input_keys,
                    prob=prob,
                    translate_range=translate_range,
                    rotate_range=(rotate_offset, rotate_offset, rotate_offset),
                    scale_range=(scale_ratio, scale_ratio, scale_ratio),
                    padding_mode="border",
                ),
                RandScaleIntensityRanged(
                    keys=input_keys, a_min=a_min_range, a_max=a_max_range, b_min=b_min, b_max=b_max, clip=True
                ),
                SpatialCropd(keys=input_keys, roi_start=roi_start, roi_end=roi_end),
                Resized(keys=output_keys, spatial_size=output_shape),
                EnsureTyped(keys=output_keys),
            ]
        )
        valid = Compose(
            [
                LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True),
                AddChanneld(keys=input_keys),
                ScaleIntensityRanged(keys=input_keys, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
                SpatialCropd(keys=input_keys, roi_start=roi_start, roi_end=roi_end),
                Resized(keys=output_keys, spatial_size=output_shape),
                EnsureTyped(keys=output_keys),
            ]
        )
    elif key == "ct-cta-ventricles":
        # http://www.ieomsociety.org/singapore2021/papers/41.pdf
        # Suggests a window of -12.5 to 82.5 would be good
        a_offset = 5
        a_min, a_max = -12.5, 82.5
        a_min_range, a_max_range = (a_min - a_offset, a_min + a_offset), (a_max - a_offset, a_max + a_offset)
        b_min, b_max = 0.0, 1.0
        image_keys = [ct_key, cta_key]
        image_and_mask_keys = [ct_key, cta_key, ct_mask_key]

        # Crop to around the ventricles
        # Crop may be able to be a tighter around the skull
        # 291, 365, 48
        roi_start = [60, 70, 57]
        roi_end = [351, 435, 105]

        # Augmentation settings
        prob = 1.0
        intensity_offset = 2  # +- 2 Hu
        # translate_range = (411 * 0.05, 493 * 0.05, 181 * 0.025)  # +-5%, +-5%, +-2.5% in each dimension X, Y, Z
        translate_range = (0, 0, 181 * 0.025)  # +-5%, +-5%, +-2.5% in each dimension X, Y, Z
        # rotate_offset = 2 * np.pi / 180  # +- 2 degrees
        rotate_offset = 0
        scale_ratio = 0.05  # +-5%

        train = Compose(
            [
                # Load CT and CTA and add a channel
                # Fill the missing image with -1000 HU to prevent KeyError: 'ct_image_transforms'
                LoadImaged(keys=image_and_mask_keys, image_only=True, allow_missing_keys=True),
                FillMissingImaged(keys=image_keys, constant=a_min, shape_key=ct_key),
                AddChanneld(keys=image_and_mask_keys),
                # Randon transforms for CT and CTA
                RandShiftIntensityd(keys=image_keys, prob=prob, offsets=intensity_offset),
                RandAffined(
                    keys=image_and_mask_keys,
                    prob=prob,
                    translate_range=translate_range,
                    rotate_range=(rotate_offset, rotate_offset, rotate_offset),
                    scale_range=(scale_ratio, scale_ratio, scale_ratio),
                    padding_mode="border",
                ),
                RandScaleIntensityRanged(
                    keys=image_keys, a_min=a_min_range, a_max=a_max_range, b_min=b_min, b_max=b_max, clip=True
                ),
                # Strip skull: has to come after ScaleIntensityRanged as it will set areas outside the mask
                # to zero, that is, b_min.
                MaskIntensityd(keys=image_keys, mask_key=ct_mask_key),
                # Crop around ventricles
                SpatialCropd(keys=image_keys, roi_start=roi_start, roi_end=roi_end),
                # Concatenate CT and CTA, delete original images and resize to shape
                ConcatItemsd(keys=image_keys, name=output_key),
                DeleteItemsd(keys=image_and_mask_keys),
                Resized(keys=output_keys, spatial_size=output_shape),
                EnsureTyped(keys=output_keys),
            ]
        )

        valid = Compose(
            [
                # Load CT and CTA and add a channel
                LoadImaged(keys=image_and_mask_keys, image_only=True, allow_missing_keys=True),
                FillMissingImaged(keys=image_keys, constant=a_min, shape_key=ct_key),
                AddChanneld(keys=image_and_mask_keys),
                # Window CT and CTA
                ScaleIntensityRanged(keys=image_keys, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
                # Strip skull: has to come after ScaleIntensityRanged as it will set areas outside the mask
                # to zero, that is, b_min.
                MaskIntensityd(keys=image_keys, mask_key=ct_mask_key),
                # Crop around ventricles
                SpatialCropd(keys=image_keys, roi_start=roi_start, roi_end=roi_end),
                # Concatenate CT and CTA, delete original images and resize to shape
                ConcatItemsd(keys=image_keys, name=output_key),
                DeleteItemsd(keys=image_and_mask_keys),
                Resized(keys=output_keys, spatial_size=output_shape),
                EnsureTyped(keys=output_keys),
            ]
        )

    return train, valid


def ventricle_roi(shape: Tuple[int, int, int]):
    """Returns ROI start and end ratios for cropping around the ventricles based on the 0.44x0.44x1.0mm images.

    :return:
    """

    roi_start = (int(60.0 / 411 * shape[0]), int(70.0 / 493 * shape[1]), int(57.0 / 181 * shape[2]))
    roi_end = (int(351.0 / 411 * shape[0]), int(435 / 493 * shape[1]), int(105.0 / 181 * shape[2]))
    return roi_start, roi_end


def make_transforms_mrs(
    *,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    ct: bool = True,
    cta: bool = False,
    ss: bool = False,
    ventricles: bool = False,
    ventricles_roi_start: List = None,
    ventricles_roi_end: List = None,
    rand: bool = False,
    rand_trans_x: float = 21,
    rand_trans_y: float = 25,
    rand_trans_z: float = 5,
    rand_rotate_offset: float = 2 * np.pi / 180,
    rand_scale_ratio: float = 0.05,
    intensity_offset: int = 2,
    dynamic_window: bool = False,
    dynamic_window_a_offset=5,
    a_min: int = -1000,  # TODO: for dynamic a_min, a_max = -12.5, 82.5
    a_max: int = 1000,
    b_min: float = 0.0,
    b_max: float = 1.0,
    resize: bool = False,
    ct_key: str = "ct_image",
    cta_key: str = "cta_image",
    output_key: str = "ct_image",
    ct_mask_key: str = "ct_mask",
    **kwargs,
):
    # Create keys
    input_keys = []
    scan_keys = []
    output_keys = [output_key]
    if ct:
        input_keys.append(ct_key)
        scan_keys.append(ct_key)
    if cta:
        input_keys.append(cta_key)
        scan_keys.append(cta_key)
    if ss:
        input_keys.append(ct_mask_key)

    # Load images
    train = [LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True)]
    valid = [LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True)]

    # If CTA is being used, fill the missing CTA with -1000 HU to prevent KeyError: 'cta_image_transforms'
    if cta:
        train.append(FillMissingImaged(keys=[cta_key], constant=a_min, shape_key=ct_key))
        valid.append(FillMissingImaged(keys=[cta_key], constant=a_min, shape_key=ct_key))

    # Add an extra channel
    train.append(AddChanneld(keys=input_keys))
    valid.append(AddChanneld(keys=input_keys))

    # Add random transforms
    if rand:
        prob = 1.0
        translate_range = (rand_trans_x, rand_trans_y, rand_trans_z)
        train.append(RandShiftIntensityd(keys=scan_keys, prob=prob, offsets=intensity_offset))
        train.append(
            RandAffined(
                keys=input_keys,
                prob=prob,
                translate_range=translate_range,
                rotate_range=(rand_rotate_offset, rand_rotate_offset, rand_rotate_offset),
                scale_range=(rand_scale_ratio, rand_scale_ratio, rand_scale_ratio),
                padding_mode="border",
            )
        )

    # If dynamic_intensity, then add RandScaleIntensityRanged which performs a random window of the scan
    # within a set range
    if dynamic_window:
        a_min_range = (a_min - dynamic_window_a_offset, a_min + dynamic_window_a_offset)
        a_max_range = (a_max - dynamic_window_a_offset, a_max + dynamic_window_a_offset)
        train.append(
            RandScaleIntensityRanged(
                keys=scan_keys, a_min=a_min_range, a_max=a_max_range, b_min=b_min, b_max=b_max, clip=True
            )
        )

    # If not dynamic_intensity, then use a standard ScaleIntensityRanged transform
    if not dynamic_window:
        train.append(
            ScaleIntensityRanged(keys=scan_keys, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True)
        )

    # For either value of dynamic_intensity, use a standard ScaleIntensityRanged transform for the validation dataset
    valid.append(ScaleIntensityRanged(keys=scan_keys, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True))

    # If ss, then strip the skull of both scans. Delete the mask after it has been used.
    if ss:
        train.append(MaskIntensityd(keys=scan_keys, mask_key=ct_mask_key))
        train.append(DeleteItemsd(keys=[ct_mask_key]))

        valid.append(MaskIntensityd(keys=scan_keys, mask_key=ct_mask_key))
        valid.append(DeleteItemsd(keys=[ct_mask_key]))

    # Crop to around the ventricles
    # Crop may be able to be a tighter around the skull
    # 291, 365, 48
    if ventricles:
        if ventricles_roi_start is None:
            ventricles_roi_start: List = [60, 70, 57]
        if ventricles_roi_end is None:
            ventricles_roi_end: List = [351, 435, 105]
        train.append(SpatialCropd(keys=scan_keys, roi_start=ventricles_roi_start, roi_end=ventricles_roi_end))
        valid.append(SpatialCropd(keys=scan_keys, roi_start=ventricles_roi_start, roi_end=ventricles_roi_end))

    # Concatenate CT and CTA together and delete original images
    if ct and cta:
        train.append(ConcatItemsd(keys=scan_keys, name=output_key))
        train.append(DeleteItemsd(keys=scan_keys))

        valid.append(ConcatItemsd(keys=scan_keys, name=output_key))
        valid.append(DeleteItemsd(keys=scan_keys))

    # Resize final output
    if resize:
        train.append(Resized(keys=output_keys, spatial_size=output_shape))
        valid.append(Resized(keys=output_keys, spatial_size=output_shape))

    train.append(EnsureTyped(keys=output_keys))
    valid.append(EnsureTyped(keys=output_keys))

    return Compose(train), Compose(valid)


def make_transforms_mrs_v2(
    *,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    ct: bool = True,
    cta: bool = False,
    ss: bool = False,
    ventricles: bool = False,
    ventricles_roi_start: List = None,
    ventricles_roi_end: List = None,
    rand_trans_x: float = 21,
    rand_trans_y: float = 25,
    rand_trans_z: float = 5,
    rand_rotate_offset: float = 2 * np.pi / 180,
    rand_scale_ratio: float = 0.05,
    intensity_offset: int = 2,
    a_min: int = -1000,
    a_max: int = 1000,
    dynamic_window_offset=5,
    resize: bool = False,
    ct_key: str = "ct_image",
    cta_key: str = "cta_image",
    output_key: str = "ct_image",
    ct_mask_key: str = "ct_mask",
    hu_mean: float = 7.836449901775156,
    hu_std: float = 32.9200692136461,
    **kwargs,
):
    # Create keys
    input_keys = []
    scan_keys = []
    output_keys = [output_key]
    affine_keys = [output_key]
    if ct:
        input_keys.append(ct_key)
        scan_keys.append(ct_key)
    if cta:
        input_keys.append(cta_key)
        scan_keys.append(cta_key)
    if ss:
        input_keys.append(ct_mask_key)
        affine_keys.append(ct_mask_key)

    # Load images
    train = [LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True)]
    valid = [LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True)]

    # If CTA is being used, fill the missing CTA with -1000 HU to prevent KeyError: 'cta_image_transforms'
    if cta:
        train.append(FillMissingImaged(keys=[cta_key], constant=-1000, shape_key=ct_key))
        valid.append(FillMissingImaged(keys=[cta_key], constant=-1000, shape_key=ct_key))

    # Add an extra channel
    train.append(AddChanneld(keys=input_keys))
    valid.append(AddChanneld(keys=input_keys))

    # Concatenate CT and CTA together and delete original images
    if ct and cta:
        train.append(ConcatItemsd(keys=scan_keys, name=output_key))
        train.append(DeleteItemsd(keys=scan_keys))

        valid.append(ConcatItemsd(keys=scan_keys, name=output_key))
        valid.append(DeleteItemsd(keys=scan_keys))

    # Clip to [a_min, a_max]
    min_val_range = (a_min - dynamic_window_offset, a_min + dynamic_window_offset)
    max_val_range = (a_max - dynamic_window_offset, a_max + dynamic_window_offset)
    train.append(RandClipd(keys=output_keys, min_val=min_val_range, max_val=max_val_range))
    valid.append(Clipd(keys=output_keys, min_val=a_min, max_val=a_max))

    aug_transforms = SomeOf(
        [
            OneOf(
                [
                    RandScaleIntensityd(keys=output_keys, prob=1, factors=0.1),
                    RandStdShiftIntensityd(keys=output_keys, prob=1, factors=0.1),
                    RandShiftIntensityd(keys=output_keys, prob=1, offsets=intensity_offset),
                    # RandHistogramShiftd(keys=scan_keys, prob=1), # causes nan loss when trained on CTA
                ]
            ),
            RandGaussianSmoothd(keys=output_keys, prob=1),
            RandGaussianSharpend(
                keys=output_keys,
                prob=1,
                sigma1_x=(0.1, 0.2),
                sigma1_y=(0.1, 0.2),
                sigma1_z=(0.1, 0.2),
                sigma2_x=8.5,
                sigma2_y=8.5,
                sigma2_z=8.5,
                alpha=(1, 2),
            ),
            RandGaussianNoised(keys=output_keys, prob=1, mean=hu_mean, std=hu_std / 4),
            RandAffined(
                keys=affine_keys,  # move CT mask too
                prob=1,
                translate_range=(rand_trans_x, rand_trans_y, rand_trans_z),
                rotate_range=(rand_rotate_offset, rand_rotate_offset, rand_rotate_offset),
                scale_range=(rand_scale_ratio, rand_scale_ratio, rand_scale_ratio),
                padding_mode="border",
            ),
        ],
        fixed=False,
        sample_max=4,
    )
    train.append(aug_transforms)

    # Strip skull
    if ss:
        train.append(MaskIntensity2d(keys=output_keys, mask_key=ct_mask_key, mask_value=a_min))
        train.append(DeleteItemsd(keys=[ct_mask_key]))
        valid.append(MaskIntensity2d(keys=output_keys, mask_key=ct_mask_key, mask_value=a_min))
        valid.append(DeleteItemsd(keys=[ct_mask_key]))

    # Crop to around the ventricles
    # Crop may be able to be a tighter around the skull
    # 291, 365, 48
    if ventricles:
        if ventricles_roi_start is None:
            ventricles_roi_start: List = [60, 70, 57]
        if ventricles_roi_end is None:
            ventricles_roi_end: List = [351, 435, 105]
        train.append(SpatialCropd(keys=output_keys, roi_start=ventricles_roi_start, roi_end=ventricles_roi_end))
        valid.append(SpatialCropd(keys=output_keys, roi_start=ventricles_roi_start, roi_end=ventricles_roi_end))

    # Resize final output
    if resize:
        train.append(Resized(keys=output_keys, spatial_size=output_shape))
        valid.append(Resized(keys=output_keys, spatial_size=output_shape))

    train.append(NormalizeIntensityd(keys=output_keys, subtrahend=hu_mean, divisor=hu_std))
    valid.append(NormalizeIntensityd(keys=output_keys, subtrahend=hu_mean, divisor=hu_std))

    train.append(EnsureTyped(keys=output_keys))
    valid.append(EnsureTyped(keys=output_keys))

    return Compose(train), Compose(valid)


def make_transforms_baseline(
    *,
    output_shape: Tuple[int, int, int],
    ct: bool = True,
    cta: bool = False,
    ss: bool = False,
    ventricles: bool = False,
    ventricles_roi_start: List = None,
    ventricles_roi_end: List = None,
    a_min: int = -1000,
    a_max: int = 1000,
    resize: bool = False,
    ct_key: str = "ct_image",
    cta_key: str = "cta_image",
    output_key: str = "ct_image",
    ct_mask_key: str = "ct_mask",
    **kwargs,
):
    # Create keys
    input_keys = []
    scan_keys = []
    output_keys = [output_key]
    if ct:
        input_keys.append(ct_key)
        scan_keys.append(ct_key)
    if cta:
        input_keys.append(cta_key)
        scan_keys.append(cta_key)
    if ss:
        input_keys.append(ct_mask_key)

    # Load images
    transforms = [LoadImaged(keys=input_keys, image_only=True, allow_missing_keys=True)]

    # If CTA is being used, fill the missing CTA with -1000 HU to prevent KeyError: 'cta_image_transforms'
    if cta:
        transforms.append(FillMissingImaged(keys=[cta_key], constant=-1000, shape_key=ct_key))

    # Add an extra channel
    transforms.append(AddChanneld(keys=input_keys))

    # Concatenate CT and CTA together and delete original images
    if ct and cta:
        transforms.append(ConcatItemsd(keys=scan_keys, name=output_key))
        transforms.append(DeleteItemsd(keys=scan_keys))

    # Clip to [a_min, a_max]
    transforms.append(Clipd(keys=output_keys, min_val=a_min, max_val=a_max))

    # Strip skull
    if ss:
        transforms.append(MaskIntensity2d(keys=output_keys, mask_key=ct_mask_key, mask_value=a_min))
        transforms.append(DeleteItemsd(keys=[ct_mask_key]))

    # Crop to around the ventricles
    # Crop may be able to be a tighter around the skull
    # 291, 365, 48
    if ventricles:
        if ventricles_roi_start is None:
            ventricles_roi_start: List = [60, 70, 57]
        if ventricles_roi_end is None:
            ventricles_roi_end: List = [351, 435, 105]
        transforms.append(SpatialCropd(keys=output_keys, roi_start=ventricles_roi_start, roi_end=ventricles_roi_end))

    # Resize final output
    if resize:
        transforms.append(Resized(keys=output_keys, spatial_size=output_shape))

    transforms.append(EnsureTyped(keys=output_keys))

    return Compose(transforms), None
