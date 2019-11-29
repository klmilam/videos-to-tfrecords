"""Feature management for data preprocessing."""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema


CATEGORICAL_COLUMNS = []
STRING_COLUMNS = ["filename"]
NUMERIC_COLUMNS = ["timestamp_ms", "frame_per_sec"]
NUMERIC_LIST_COLUMNS =["image"]
BOOLEAN_COLUMNS = []

def get_raw_feature_spec():
    """Returns TF feature spec for preprocessing"""
    features = dict(
        [(name, tf.io.FixedLenFeature([], tf.string))
            for name in CATEGORICAL_COLUMNS] +
        [(name, tf.io.FixedLenFeature([], tf.string))
            for name in STRING_COLUMNS] +
        [(name, tf.io.FixedLenFeature([], tf.float32))
            for name in NUMERIC_COLUMNS] +
        [(name, tf.io.FixedLenFeature([], tf.int64))
            for name in BOOLEAN_COLUMNS] +
        [(name, tf.io.FixedLenFeature([], tf.float32))
            for name in NUMERIC_LIST_COLUMNS]
    )
    return features


RAW_FEATURE_SPEC = get_raw_feature_spec()

def preprocess(inputs):
    return inputs.copy()