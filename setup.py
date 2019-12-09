# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Config for Dataflow preprocessing."""

from setuptools import find_packages
from setuptools import setup


NAME = "Videos-To-TFRecords"
VERSION = "1.0"
REQUIRED_PACKAGES = ["opencv-python",
                     "google-cloud-storage",
                     "google-resumable-media==0.5.0",
                     "tensorflow_hub>=0.6.0",
                     "tensorflow==2.0.0",
                     "tensorflow-transform"]

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
)
