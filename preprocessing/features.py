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
import logging


FLOAT = "float"
INT = "int"
BYTES = "bytes"
OTHER = ""

LIST_COLUMNS = {
    "logits": FLOAT,
    "timestamp_ms": FLOAT
}

CONTEXT_COLUMNS = {
    "label": BYTES,
    "filename": BYTES,
    "dataset": BYTES,
    "frame_per_sec": FLOAT,
    "frame_total": FLOAT
}


def to_feature_list(value, dtype):
    """Wraps feature in a TF.Feature protocol message."""
    if not isinstance(value, list):
        value = [value]  # values must be lists
    if dtype == FLOAT:
        # TODO: add error catching if not float (or not able to be cast to float)
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    elif dtype == INT:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    elif dtype == BYTES:
        if type(value[0]) == str:  # all elements in list should have same type
            byte_values = []
            for v in value:
                byte_values.append(str.encode(v))
            value = byte_values
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        logging.warning("Type {} not supported.".format(type(value[0])))
