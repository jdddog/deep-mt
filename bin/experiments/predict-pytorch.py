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
import os
import os.path
from typing import List, Tuple

from deep_mt.config import Config
from deep_mt.predict import predict


def main(data_path: str, configs: List[Tuple[str, str]]):
    for config_path, weights_name in configs:
        config_path = os.path.join(data_path, config_path)
        config = Config.load(config_path)
        weights_path = os.path.join(config.experiment_folder, weights_name)
        predict(config, weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config folder")
    args = parser.parse_args()

    paths = [
        # Sex
        (
            "imaging/sex/sex-ct-152x182x76px.yaml",
            "sex-ct-152x182x76px_epoch_83.pth",
        ),
        (
            "imaging/sex/sex-ct-1.0x1.0x2.0mm-152x182x76px.yaml",
            "sex-ct-1.0x1.0x2.0mm-152x182x76px_epoch_30.pth",
        ),
        (
            "imaging/sex/sex-ct-ss-1.0x1.0x2.0mm-152x182x76px.yaml",
            "sex-ct-ss-1.0x1.0x2.0mm-152x182x76px_epoch_33.pth",
        ),
        (
            "imaging/sex/sex-ct-ss-1.0x1.0x2.0mm-152x182x76px-rand-trans.yaml",
            "sex-ct-ss-1.0x1.0x2.0mm-152x182x76px-rand-trans_epoch_82.pth",
        ),
        # mRS: 1.0x1.0x2.0mm
        (
            "imaging/mrs/mrs-ct-1.0x1.0x2.0mm-152x182x76px.yaml",
            "mrs-ct-1.0x1.0x2.0mm-152x182x76px_epoch_33.pth",
        ),
        (
            "imaging/mrs/mrs-ct-cta-1.0x1.0x2.0mm-152x182x76px.yaml",
            "mrs-ct-cta-1.0x1.0x2.0mm-152x182x76px_epoch_18.pth",
        ),
        (
            "imaging/mrs/mrs-ct-1.0x1.0x2.0mm-152x182x76px-no-basilars.yaml",
            "mrs-ct-1.0x1.0x2.0mm-152x182x76px-no-basilars_epoch_59.pth",
        ),
        (
            "imaging/mrs/mrs-ct-cta-1.0x1.0x2.0mm-152x182x76px-no-basilars.yaml",
            "mrs-ct-cta-1.0x1.0x2.0mm-152x182x76px-no-basilars_epoch_72.pth",
        ),
        # mRS: ventricles
        (
            "imaging/mrs/mrs-ct-ventricles-0.44x0.44x1.5mm.yaml",
            "mrs-ct-ventricles-0.44x0.44x1.5mm_epoch_82.pth",
        ),
        (
            "imaging/mrs/mrs-ct-cta-ventricles-0.44x0.44x1.5mm.yaml",
            "mrs-ct-cta-ventricles-0.44x0.44x1.5mm_epoch_42.pth",
        ),
        # mRS: combined
        (
            "combined/mrs-ct-1.0x1.0x2.0mm-152x182x76px-fine-tune-squeeze.yaml",
            "mrs-ct-1.0x1.0x2.0mm-152x182x76px-fine-tune-squeeze_epoch_123.pth",
        ),
        (
            "combined/mrs-ct-1.0x1.0x2.0mm-152x182x76px-fine-tune-squeeze-all-features.yaml",
            "mrs-ct-1.0x1.0x2.0mm-152x182x76px-fine-tune-squeeze-all-features_epoch_82.pth",
        ),
        (
            "combined/mrs-ct-1.0x1.0x2.0mm-152x182x76px-fine-tune-squeeze-no-basilars.yaml",
            "mrs-ct-1.0x1.0x2.0mm-152x182x76px-fine-tune-squeeze-no-basilars_epoch_149.pth",
        ),
        (
            "combined/mrs-ct-cta-1.0x1.0x2.0mm-152x182x76px-fine-tune-squeeze-no-basilars-epoch-21.yaml",
            "mrs-ct-cta-1.0x1.0x2.0mm-152x182x76px-fine-tune-squeeze-no-basilars-epoch-21_epoch_47.pth",
        ),
    ]
    main(args.data_path, paths)
