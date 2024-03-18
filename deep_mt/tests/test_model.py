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

import unittest

import torch
from monai.networks.nets import DenseNet121

from deep_mt.ml_utils import make_model
from deep_mt.monai_utils import set_determinism


def compare_weights(model_a, model_b):
    same = True
    # Assert that weights are the same
    for param1, param2 in zip(model_a.parameters(), model_b.parameters()):
        if not torch.equal(param1, param2):
            print("Models have different weights")
            same = False
            break
    else:
        print("Models have the same weights")

    return same


class TestModel(unittest.TestCase):
    def test_make_model(self):
        """Check that make_model creates a model and that the weights are the same using set_determinism"""

        seed = 7
        class_name = "monai.networks.nets.DenseNet121"
        kwargs = {"spatial_dims": 3, "in_channels": 1, "out_channels": 2}

        # Assert that models return DenseNet121
        set_determinism(seed=seed)
        model1 = make_model(class_name, kwargs)
        self.assertIsInstance(model1, DenseNet121)

        set_determinism(seed=seed)
        model2 = make_model(class_name, kwargs)
        self.assertIsInstance(model2, DenseNet121)

        # Assert that models have same weights
        self.assertTrue(compare_weights(model1, model2))

        # Assert that models have different weights
        model3 = make_model(class_name, kwargs)
        self.assertIsInstance(model3, DenseNet121)
        self.assertFalse(compare_weights(model2, model3))
