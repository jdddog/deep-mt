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

import datetime
import os
import re
from timeit import default_timer as timer


def print_duration(func):
    """Print the duration of a function.

    :param func: the function to execute.
    :return: the wrapper.
    """

    def wrapper(*args, **kwargs):
        # Get start time
        start = timer()

        # Run function
        func(*args, **kwargs)

        # Calculate and print duration
        end = timer()
        duration = end - start
        print(f"Duration: {datetime.timedelta(seconds=duration)}")

    return wrapper


def match_files(path: str, pattern: str):
    pattern = re.compile(pattern)
    filtered_files = []

    for root, dirs, files in os.walk(path):
        for f in files:
            if pattern.match(f):
                filtered_files.append(os.path.join(root, f))

    return filtered_files
