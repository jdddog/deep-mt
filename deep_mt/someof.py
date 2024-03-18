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

import random
from typing import Callable, Optional, Sequence, Union

import numpy as np
from monai.transforms.compose import Compose


class SomeOf(Compose):
    """
    ``SomeOf`` provides the ability to randomly sample a fixed number or
    up to a fixed number of transforms from a list of callables.

    Derivse from the `Compose` class.

    Args:
        transforms: list of callables.
        sample_max: maximum number to sample. defaults to `3`.
        fixed: whether to sample a fixed number of transforms (sample_max). defaults to `False`.
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            defaults to `True`.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            defaults to `False`.
        log_stats: whether to log the detailed information of data and applied transform when error happened,
            for NumPy array and PyTorch Tensor, log the data shape and value range,
            for other metadata, log the values directly. default to `False`.

    """

    def __init__(
        self,
        transforms: Optional[Union[Sequence[Callable], Callable]] = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool = False,
        sample_max: int = 3,
        fixed: bool = True,
    ) -> None:
        super().__init__(transforms, map_items, unpack_items, log_stats)
        self.original_transforms = transforms
        self.n_transforms = len(transforms)
        self.sample_max = min(self.n_transforms, sample_max)
        self.fixed = fixed
        self._update_active_transforms()

    def _update_active_transforms(self):
        if self.n_transforms == 0:
            return

        sample_size = self.sample_max if self.fixed else random.randint(1, self.sample_max)
        self.active = random.sample(np.arange(self.n_transforms).tolist(), sample_size)
        self.transforms = [self.original_transforms[i] for i in self.active]

    def __call__(self, data):  # Override Compose
        result = super().__call__(data)
        self._update_active_transforms()
        return result
