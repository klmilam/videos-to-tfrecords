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
"""Feature management for data preprocessing."""

import tensorflow as tf


FLOAT = "float"
INT = "int"
BYTES = "bytes"

LIST_COLUMNS = {
    "logits": FLOAT,
    "timestamp_ms": FLOAT
}


def to_feature_list(value, dtype):
    if not isinstance(value, list):
        value = [value]
    if dtype == FLOAT:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    elif dtype == INT:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    elif dtype == BYTES:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
