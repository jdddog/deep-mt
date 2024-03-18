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

from typing import Callable, Dict, Hashable, List, Mapping, Optional, Tuple

import numpy as np
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms import (MapTransform, Randomizable, RandomizableTransform, ScaleIntensityRanged)
from monai.transforms.transform import Transform
from monai.transforms.utils_pytorch_numpy_unification import clip
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor


class DropMissingKeyd(MapTransform):
    def __init__(self, image_exists_keys: KeysCollection, keys_to_drop: List) -> None:
        MapTransform.__init__(self, image_exists_keys, allow_missing_keys=False)
        self.key_index = {}
        for k, v in zip(image_exists_keys, keys_to_drop):
            self.key_index[k] = v

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        # Drop image keys that shouldn't exist
        for key in self.key_iterator(d):
            if not d[key]:
                d.pop(self.key_index[key])
                break

        return d


class RandDropKeyd(MapTransform, RandomizableTransform):
    def __init__(self, keys: KeysCollection, prob: float = 1.0, do_transform: bool = True) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=True)
        RandomizableTransform.__init__(self, prob=prob, do_transform=do_transform)
        self._drop_index = 0
        self._max_index = len(keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        self.randomize(data)
        d = dict(data)

        # If don't do transform or all keys don't exist then return data as is
        all_keys_exist = all([key in d for key in self.keys])
        if not self._do_transform or not all_keys_exist:
            return d

        # Get the key to drop, self._drop_index randomly chosen in randomize
        # Remove the kv pair by popping
        drop_index = self.R.randint(2)
        key = self.keys[drop_index]
        # worker_info = torch.utils.data.get_worker_info()
        # print(f"Drop: {worker_info.id} {d['case_id']}: {key}")
        d.pop(key)

        return d


class FillMissingImaged(MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, constant: float, shape_key: str, dtype: DtypeLike = np.float32) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=True)
        self.shape_key = shape_key
        self.constant = constant
        self.dtype = dtype

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        # Get image shape of image specified by self.shape_key
        if self.shape_key not in d:
            raise Exception(f"Cannot find shape_key={self.shape_key} in data")
        shape = d[self.shape_key].shape

        # For all keys that don't exist, create random gaussian noise in the same shape as the expected image
        for key in self.keys:
            if key not in d:
                image = np.full(shape, self.constant, dtype=self.dtype)
                image, *_ = convert_data_type(image, dtype=self.dtype)
                d[key] = image

        return d


class FillMissingRandGaussianNoised(MapTransform, Randomizable):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self, keys: KeysCollection, shape: Tuple, mean: float = 0.0, std: float = 0.1, dtype: DtypeLike = np.float32
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys=True)
        Randomizable.__init__(self)
        self.shape = shape
        self.mean = mean
        self.std = std
        self.dtype = dtype
        self.noise: Optional[np.ndarray] = None

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        # For all keys that don't exist, create random gaussian noise in the same shape as the expected image
        for key in self.keys:
            if key not in d:
                noise = self.R.normal(self.mean, self.std, size=self.shape)
                noise, *_ = convert_data_type(noise, dtype=self.dtype)
                d[key] = noise

        return d


class RandScaleIntensityRanged(Randomizable):
    def __init__(
        self,
        keys: KeysCollection,
        a_min: Tuple[float, float],
        a_max: Tuple[float, float],
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        clip: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        Randomizable.__init__(self)
        self.keys = keys
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.dtype = dtype
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        # Calculate random a_min and a_max within a range
        a_min = self.R.randint(low=self.a_min[0], high=self.a_min[1])
        a_max = self.R.randint(low=self.a_max[0], high=self.a_max[1])

        # Scale data
        scaler = ScaleIntensityRanged(
            keys=self.keys,
            a_min=a_min,
            a_max=a_max,
            b_min=self.b_min,
            b_max=self.b_max,
            clip=self.clip,
            allow_missing_keys=self.allow_missing_keys,
        )
        return scaler(data)


class Clip(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, *, min_val, max_val, dtype=np.float32):
        self.min_val = min_val
        self.max_val = max_val
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor):
        dtype = self.dtype or img.dtype
        img = clip(img, self.min_val, self.max_val)
        ret: NdarrayOrTensor = convert_data_type(img, dtype=dtype)[0]
        return ret


class Clipd(MapTransform):
    backend = Clip.backend

    def __init__(
        self,
        keys: KeysCollection,
        min_val: float,
        max_val: float,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.clipper = Clip(min_val=min_val, max_val=max_val, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.clipper(d[key])
        return d


class RandClipd(Randomizable, MapTransform):
    backend = Clip.backend

    def __init__(
        self,
        keys: KeysCollection,
        min_val: Tuple[float, float],
        max_val: Tuple[float, float],
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        Randomizable.__init__(self)
        MapTransform.__init__(self, keys=keys, allow_missing_keys=allow_missing_keys)
        self.keys = keys
        self.min_val = min_val
        self.max_val = max_val
        self.dtype = dtype
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        # Calculate random a_min and a_max within a range
        min_val = self.R.randint(low=self.min_val[0], high=self.min_val[1])
        max_val = self.R.randint(low=self.max_val[0], high=self.max_val[1])
        clipper = Clip(min_val=min_val, max_val=max_val, dtype=self.dtype)

        # Scale data
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = clipper(d[key])
        return d


class MaskIntensity2(Transform):
    """
    Mask the intensity values of input image with the specified mask data.
    Mask data must have the same spatial size as the input image, and all
    the intensity values of input image corresponding to the selected values
    in the mask data will keep the original value, others will be set to `0`.

    Args:
        mask_data: if `mask_data` is single channel, apply to every channel
            of input image. if multiple channels, the number of channels must
            match the input data. the intensity values of input image corresponding
            to the selected values in the mask data will keep the original value,
            others will be set to `0`. if None, must specify the `mask_data` at runtime.
        select_fn: function to select valid values of the `mask_data`, default is
            to select `values > 0`.
        mask_value: masked value to OR with the image.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self, mask_data: Optional[NdarrayOrTensor] = None, select_fn: Callable = lambda img: img <= 0, mask_value=-1e5
    ) -> None:
        self.mask_data = mask_data
        self.select_fn = select_fn
        self.mask_value = mask_value

    def __call__(self, img: NdarrayOrTensor, mask_data: Optional[NdarrayOrTensor] = None) -> NdarrayOrTensor:
        """
        Args:
            mask_data: if mask data is single channel, apply to every channel
                of input image. if multiple channels, the channel number must
                match input data. mask_data will be converted to `bool` values
                by `mask_data > 0` before applying transform to input image.

        Raises:
            - ValueError: When both ``mask_data`` and ``self.mask_data`` are None.
            - ValueError: When ``mask_data`` and ``img`` channels differ and ``mask_data`` is not single channel.

        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        mask_data = self.mask_data if mask_data is None else mask_data
        if mask_data is None:
            raise ValueError("must provide the mask_data when initializing the transform or at runtime.")

        mask_data_, *_ = convert_to_dst_type(src=mask_data, dst=img)
        mask_data_ = self.select_fn(mask_data_)
        img[mask_data_] = self.mask_value

        if mask_data_.shape[0] != 1 and mask_data_.shape[0] != img.shape[0]:
            raise ValueError(
                "When mask_data is not single channel, mask_data channels must match img, "
                f"got img channels={img.shape[0]} mask_data channels={mask_data_.shape[0]}."
            )

        return convert_to_dst_type(img, dst=img)[0]


class MaskIntensity2d(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MaskIntensity`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        mask_data: if mask data is single channel, apply to every channel
            of input image. if multiple channels, the channel number must
            match input data. the intensity values of input image corresponding
            to the selected values in the mask data will keep the original value,
            others will be set to `0`. if None, will extract the mask data from
            input data based on `mask_key`.
        mask_key: the key to extract mask data from input dictionary, only works
            when `mask_data` is None.
        select_fn: function to select valid values of the `mask_data`, default is
            to select `values > 0`.
        allow_missing_keys: don't raise exception if key is missing.
        mask_value: masked value to OR with the image.
    """

    backend = MaskIntensity2.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_data: Optional[NdarrayOrTensor] = None,
        mask_key: Optional[str] = None,
        select_fn: Callable = lambda img: img <= 0,
        allow_missing_keys: bool = False,
        mask_value: int = -1e5,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = MaskIntensity2(mask_data=mask_data, select_fn=select_fn, mask_value=mask_value)
        self.mask_key = mask_key if mask_data is None else None

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key], d[self.mask_key]) if self.mask_key is not None else self.converter(d[key])
        return d
