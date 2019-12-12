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
"""Build preprocessing pipeline."""

import apache_beam as beam
from apache_beam.transforms import combiners
import tensorflow as tf
import numpy as np
import logging
import io
import os
import tempfile
import cv2
import datetime
import urllib
from google.cloud import storage
import tensorflow_hub as hub
import random

from preprocessing import features


@beam.ptransform_fn
def randomly_split(p, train_size, validation_size, test_size):
    """Randomly splits input pipeline in three sets based on input ratio.
    Args:
        p: PCollection, input pipeline.
        train_size: float, ratio of data going to train set.
        validation_size: float, ratio of data going to validation set.
        test_size: float, ratio of data going to test set.
    Returns:
        Tuple of PCollection.
    Raises:
        ValueError: Train validation and test sizes don`t add up to 1.0.
    """
    if train_size + validation_size + test_size != 1.0:
        raise ValueError(
            "Train validation and test sizes don`t add up to 1.0.")

    class _SplitData(beam.DoFn):
        def process(self, element):
            r = random.random()
            if r < test_size:
                element["dataset"] = "Test"
            elif r < 1 - train_size:
                element["dataset"] = "Val"
            else:
                element["dataset"] = "Train"
            yield element

    split_data = p | "SplitData" >> beam.ParDo(_SplitData())
    return split_data


@beam.ptransform_fn
def shuffle(p):
    """Shuffles the given pCollection."""
    return (p
            | 'PairWithRandom' >> beam.Map(lambda x: (random.random(), x))
            | 'GroupByRandom' >> beam.GroupByKey()
            | 'DropRandom' >> beam.FlatMap(lambda x: x[1]))


def generate_seq_example(element):
    feature_list_dict = {}
    for feat, dtype in features.LIST_COLUMNS.items():
        feature_list = []
        for value in element[feat]:
            feature_list.append(
                features.to_feature_list(value, dtype))
        feature_list_dict[feat] = tf.train.FeatureList(feature=feature_list)

    context_dict = {}
    for feat, dtype in features.CONTEXT_COLUMNS.items():
        context_dict[feat] = features.to_feature_list(
            element[feat], dtype)

    seq_example = tf.train.SequenceExample(
        context=tf.train.Features(feature=context_dict),
        feature_lists=tf.train.FeatureLists(feature_list=feature_list_dict))
    return seq_example


@beam.ptransform_fn
def WriteTFRecord(p, prefix, output_dir):
    """Shuffles and write the given pCollection as a TF-Record.
    Args:
        p: a pCollection.
        prefix: prefix for location TFRecord will be written to.
        output_dir: the directory or bucket to write the json data.
    """
    coder = beam.coders.ProtoCoder(tf.train.SequenceExample)
    prefix = str(prefix).lower()
    out_dir = os.path.join(output_dir, 'data', prefix, prefix)
    logging.warning("writing TFrecords to "+ out_dir)
    (
        p
        | "ShuffleData" >> shuffle()  # pylint: disable=no-value-for-parameter
        | "WriteTFRecord" >> beam.io.tfrecordio.WriteToTFRecord(
            os.path.join(output_dir, 'data', prefix, prefix),
            coder=coder,
            file_name_suffix=".tfrecord"))


def generate_download_signed_url_v4(service_account_file, bucket_name,
                                    blob_name):
    """Generates a v4 signed URL for downloading a blob.

    To use OpenCV's VideoCapture method, video files must be available either
    at a local directory or at a public URL. This function creates signed URLs
    to access video files in GCS.

    The service account key is copied locally so that it is accessible to the
    Storage client.
    """
    local_key = tempfile.NamedTemporaryFile(suffix=".json").name
    tf.io.gfile.copy(service_account_file, local_key)
    storage_client = storage.Client.from_service_account_json(local_key)
    os.remove(local_key)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version='v4',
        expiration=datetime.timedelta(minutes=15),
        method='GET')
    return url


class VideoToFrames(beam.DoFn):
    """Transform to read a video file from GCS and extract frames."""
    def __init__(self, service_account_file, skip_msec):
        self.service_account_file = service_account_file
        self.skip_msec = skip_msec

    def process(self, element, cloud=True):
        u = urllib.parse.urlparse(element["filename"])
        signed_url = generate_download_signed_url_v4(
            self.service_account_file, u.netloc, u.path[1:])
        video = cv2.VideoCapture(signed_url)

        last_ts = -9999
        result, image = video.read()
        limit_local = 0
        while(video.isOpened() and limit_local < 3):
            # Only record frames occurring every skip_msec
            while video.get(cv2.CAP_PROP_POS_MSEC) < self.skip_msec + last_ts:
                result, image = video.read()
                if not result:
                    return
            last_ts = video.get(cv2.CAP_PROP_POS_MSEC)
            image = image/255.  # Normalize
            image = image[:, :, ::-1]  # OpenCV orders channels BGR
            image = image[np.newaxis, :, :, :]  # Add batch dimension
            output = element.copy()
            output["image"] = image
            output["timestamp_ms"] = last_ts
            output["frame_per_sec"] = round(video.get(cv2.CAP_PROP_FPS))
            output["frame_total"] = video.get(cv2.CAP_PROP_FRAME_COUNT)
            limit_local = limit_local + 1 if not cloud else 0
            yield output
        video.release()
        cv2.destroyAllWindows()


class Inception(beam.DoFn):
    """Transform to extract Inception-V3 bottleneck features."""
    def __init__(self):
        self._model = None
        self.initialized = False

    def initialize(self):
        """Initializes the model on the workers."""
        inputs = tf.keras.Input(shape=(None, None, 3))
        inception_layer = hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
            output_shape=2048,
            trainable=False
        )
        output = inception_layer(inputs)
        model = tf.keras.Model(inputs, output)
        self._model = model
        self.initialized = True

    def process(self, element):
        if not self.initialized:
            logging.info("Initializing model.")
            self.initialize()
        logits = self._model.predict(element['image'])
        del element['image']
        element['logits'] = np.squeeze(logits).tolist()
        yield element


def extract_label(element):
    """Extracts and appends label from filename.

    Assumes there's one label per video.
    """
    element["label"] = element["filename"].split("/")[-3]
    return element


class AddTimestamp(beam.DoFn):
    def process(self, element):
        yield beam.window.TimestampedValue(element, element["timestamp_ms"])


class SetWindowVideoAsKey(beam.DoFn):
    """Transform to extract the window and set it as the key."""
    def process(self, element, sequence_length, window=beam.DoFn.WindowParam):
        """Sets an element's key as its key.
        Args:
            element: processing element (dict).
            window: the window that the element belongs to.
        Yields:
            key-value pair of a window and the input element, respectively.
        """
        video_length = 1000 * element["frame_total"] / element["frame_per_sec"]
        if float(window.end) == sequence_length or float(
            window.start) >= 0 and float(window.end) <= video_length:
            yield ((window, element["filename"]), element)


class FormatFeatures(beam.DoFn):
    def process(self, element):
        # TODO: support per-frame labels
        per_sample_output = {
            key:element[0][key] for key in features.CONTEXT_COLUMNS}

        per_frame_output = {
            key:[d[key] for d in element] for key in features.LIST_COLUMNS}

        output = {**per_sample_output, **per_frame_output}
        return [output]


@beam.ptransform_fn
def create_filenames(p, files):
    filenames = (
        p
        | "CreateFilePattern" >> beam.Create(files)
        | "CreateDict" >> beam.Map(lambda x: {"filename": x})
        # TODO: compare filenames' suffix to list of supported video suffix types
        | "FilterVideos" >> beam.Filter(
            lambda x: x["filename"].split(".")[-1] == "mkv" and
                x["filename"].split("/")[-2] == "360P")
    )
    return filenames


@beam.ptransform_fn
def crop_video(p, args):
    period = args.period if args.period else args.sequence_length
    frames = (
        p
        | "AddTimestamp" >> beam.ParDo(AddTimestamp())
        | "ApplySlidingWindow" >> beam.WindowInto(
            beam.window.SlidingWindows(args.sequence_length, period))
        | "AddWindowAndVideoAsKey" >> beam.ParDo(
            SetWindowVideoAsKey(), args.sequence_length)
        | "GroupByKey" >> beam.GroupByKey()
        | "CombineToList" >> beam.CombinePerKey(
            combiners.ToListCombineFn())
        | "ApplyGlobalWindow" >> beam.WindowInto(
            beam.window.GlobalWindows())
        | "UnKey" >> beam.Map(lambda x: x[1][0]))
    return frames


@beam.ptransform_fn
def to_full_video(p, args):
    frames = (
        p
        | "SetVideoAsKey" >> beam.Map(lambda x: (x["filename"], x))
        | "GroupByKey" >> beam.GroupByKey()
        | "CombineToList" >> beam.CombinePerKey(
            combiners.ToListCombineFn())
        | "UnKey" >> beam.Map(lambda x: x[1][0]))
    return frames


def build_pipeline(p, args):
    if args.cloud:
        path = os.path.join(args.input_dir, "*", "*", "*")
    else:
        path = os.path.join(args.input_dir, "*")
    files = tf.io.gfile.glob(path)

    filenames = (
        p
        | "CreateFilenames" >> create_filenames(files)
        | "RandomlySplitData" >> randomly_split(
            train_size=.7, validation_size=.15, test_size=.15)
        | "ExtractLabel" >> beam.Map(extract_label))

    frames = (
        filenames
        | "ExtractFrames" >> beam.ParDo(VideoToFrames(
            args.service_account_key_file, args.frame_sample_rate),  args.cloud)
        | "ApplyInception" >> beam.ParDo(Inception()))
    
    if args.mode == "crop_video":
        frames = frames | "CropVideo" >> crop_video(args)
    elif args.mode == "full_video":
        frames = frames | "ToFullVideo" >> to_full_video(args) 
    else:
        frames = frames | "ToSingleFrame" >> beam.Map(lambda x: [x])
    
    all_frames = (
        frames
        | "SortFrames" >> beam.Map(
            lambda x: sorted(x, key=lambda i: i["timestamp_ms"]))
        | "ListDictsToDictLists" >> beam.ParDo(FormatFeatures()))

    for dataset_type in ['Train', 'Val', 'Test']:
        dataset = (
            all_frames
            | "Get{}Data".format(dataset_type) >> beam.Filter(
                lambda x: x["dataset"] == dataset_type)
            | "Convert{}ToSeqExamples".format(dataset_type) >> beam.Map(
                generate_seq_example))
        if not args.cloud:  # if running locally, print SequenceExamples
            dataset | "p{}".format(dataset_type) >> beam.Map(print)
        write_label = 'Write{}TFRecord'.format(dataset_type)
        dataset | write_label >> WriteTFRecord(dataset_type, args.output_dir)
