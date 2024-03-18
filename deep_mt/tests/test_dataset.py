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

import pandas as pd

from deep_mt.dataset import centre_split, mrs_quantize, mRSType, standard_split


def make_data(num_cases: int = 100):
    data = []
    for i in range(num_cases):
        # 1 in 2 records mrs = 1
        mrs = 0
        if i % 2 == 0:
            mrs = 1

        # 1 in 3 records from ChCh
        centre = "Akl"
        if i % 3 == 0:
            centre = "ChCh"

        data.append({"mrs": mrs, "case_id": f"case{i}", "centre": centre})
    return pd.DataFrame(data)


class TestDataset(unittest.TestCase):
    def test_mrs_quantize(self):
        # mRSType.mrs02_36:
        # mRS 0-2
        expected = 0
        self.assertEqual(mrs_quantize(0, mRSType.mrs02_36), expected)
        self.assertEqual(mrs_quantize(1, mRSType.mrs02_36), expected)
        self.assertEqual(mrs_quantize(2, mRSType.mrs02_36), expected)
        # mRS 3-6
        expected = 1
        self.assertEqual(mrs_quantize(3, mRSType.mrs02_36), expected)
        self.assertEqual(mrs_quantize(4, mRSType.mrs02_36), expected)
        self.assertEqual(mrs_quantize(5, mRSType.mrs02_36), expected)
        self.assertEqual(mrs_quantize(6, mRSType.mrs02_36), expected)

        # mRSType.mrs0-6
        for i in range(0, 7):
            self.assertEqual(mrs_quantize(i, mRSType.mrs0_6), i)

        # mRSType.mrs03-46
        # mRS 0-3
        expected = 0
        self.assertEqual(mrs_quantize(0, mRSType.mrs03_46), expected)
        self.assertEqual(mrs_quantize(1, mRSType.mrs03_46), expected)
        self.assertEqual(mrs_quantize(2, mRSType.mrs03_46), expected)
        self.assertEqual(mrs_quantize(3, mRSType.mrs03_46), expected)
        # mRS 4-6
        expected = 1
        self.assertEqual(mrs_quantize(4, mRSType.mrs03_46), expected)
        self.assertEqual(mrs_quantize(5, mRSType.mrs03_46), expected)
        self.assertEqual(mrs_quantize(6, mRSType.mrs03_46), expected)

        # mRSType.mrs05_6
        # mRS 0-5
        expected = 0
        self.assertEqual(mrs_quantize(0, mRSType.mrs05_6), expected)
        self.assertEqual(mrs_quantize(1, mRSType.mrs05_6), expected)
        self.assertEqual(mrs_quantize(2, mRSType.mrs05_6), expected)
        self.assertEqual(mrs_quantize(3, mRSType.mrs05_6), expected)
        self.assertEqual(mrs_quantize(4, mRSType.mrs05_6), expected)
        self.assertEqual(mrs_quantize(5, mRSType.mrs05_6), expected)
        # mRS 6
        expected = 1
        self.assertEqual(mrs_quantize(6, mRSType.mrs05_6), expected)

        # Test assertions
        with self.assertRaises(Exception):
            mrs_quantize(-1, mRSType.mrs02_36)

        with self.assertRaises(Exception):
            mrs_quantize(7, mRSType.mrs02_36)

    def test_centre_split(self):
        # Create cases
        num_cases = 100
        df = make_data(num_cases)
        case_id_key = "case_id"
        valid_ratio = 0.2
        stratify_class = "mrs"
        centre_test_set = "ChCh"
        n_test = 34
        n_valid = num_cases * valid_ratio
        n_train = num_cases - n_test - n_valid
        random_seed = 7

        # Split dataset
        df_train, df_valid, df_test = centre_split(
            df=df,
            centre_test_set=centre_test_set,
            valid_ratio=valid_ratio,
            stratify_class=stratify_class,
            random_seed=random_seed,
        )

        # Check expected lengths
        self.assertEqual(n_train, len(df_train))
        self.assertEqual(n_valid, len(df_valid))
        self.assertEqual(n_test, len(df_test))

        # Check that all cases were included. If they are all included then there can't be duplicate cases in
        # multiple subsets
        case_ids_train = df_train[case_id_key].tolist()
        case_ids_valid = df_valid[case_id_key].tolist()
        case_ids_test = df_test[case_id_key].tolist()
        case_ids = case_ids_train + case_ids_valid + case_ids_test
        self.assertEqual(len(df), len(case_ids))

        # Check that we get the same results after splitting again
        df_train_2, df_valid_2, df_test_2 = centre_split(
            df=df,
            centre_test_set=centre_test_set,
            valid_ratio=valid_ratio,
            stratify_class=stratify_class,
            random_seed=random_seed,
        )
        self.assertEqual(df_train[case_id_key].tolist(), df_train_2[case_id_key].tolist())
        self.assertEqual(df_valid[case_id_key].tolist(), df_valid_2[case_id_key].tolist())
        self.assertEqual(df_test[case_id_key].tolist(), df_test_2[case_id_key].tolist())

    def test_standard_split(self):
        # Create cases
        num_cases = 100
        df = make_data(num_cases)
        case_id_key = "case_id"
        random_seed = 7

        # Split dataset
        train_ratio = 0.6
        valid_ratio = 0.2
        test_ratio = 0.2
        stratify_class = "mrs"
        df_train, df_valid, df_test = standard_split(
            df=df,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            stratify_class=stratify_class,
            random_seed=random_seed,
        )

        # Check that all cases were included. If they are all included then there can't be duplicate cases in
        # multiple subsets
        case_ids_train = df_train[case_id_key].tolist()
        case_ids_valid = df_valid[case_id_key].tolist()
        case_ids_test = df_test[case_id_key].tolist()
        case_ids = case_ids_train + case_ids_valid + case_ids_test
        self.assertEqual(len(df), len(case_ids))

        # Check that we get the same results after splitting again
        df_train_2, df_valid_2, df_test_2 = standard_split(
            df=df,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            stratify_class=stratify_class,
            random_seed=random_seed,
        )
        self.assertEqual(df_train[case_id_key].tolist(), df_train_2[case_id_key].tolist())
        self.assertEqual(df_valid[case_id_key].tolist(), df_valid_2[case_id_key].tolist())
        self.assertEqual(df_test[case_id_key].tolist(), df_test_2[case_id_key].tolist())
