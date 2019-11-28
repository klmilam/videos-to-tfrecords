"""Build preprocessing pipeline."""

import apache_beam as beam
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

from preprocessing import features

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
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version='v4',
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=15),
        # Allow GET requests using this URL.
        method='GET')
    return url


class GetFilenames(beam.DoFn):
    """Transform to list contents of directory recursively."""
    def process(self, path):
        """Returns contents of every directory.
        Args:
            path: path to top-level of input data directory.
 
        Returns:
            One 3-tuple for each directory of format (pathname of directory,
            list of its subdirectories, list of its files)
        """
        path = os.path.join(path, "*", "*", "*")
        return tf.io.gfile.glob(path)


class VideoToFrames(beam.DoFn):
    """Transform to read a video file from GCS and extract frames."""
    def __init__(self, service_account_file):
        self.service_account_file = service_account_file

    def process(self, filename):
        u = urllib.parse.urlparse(filename)
        signed_url = generate_download_signed_url_v4(
            self.service_account_file, u.netloc, u.path[1:])
        input_video = cv2.VideoCapture(signed_url)
        result, image = input_video.read()
        # TODO: test without resizing
        image = cv2.resize(
            image, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        image = image/255.
        image = image[:, :, ::-1]  # OpenCV orders channels BGR
        image = image[np.newaxis, :, :, :]  # Add batch dimension
        input_video.release()
        output = {
            'image': image,
            'filename': filename,
        }
        yield output


class Inception(beam.DoFn):
    """Transform to extract Inception-V3 bottleneck features."""
    def process(self, element):
        inputs = tf.keras.Input(shape=(299, 299, 3))
        inception_layer = hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
            output_shape=2048,
            trainable=False
        )
        output = inception_layer(inputs)
        m = tf.keras.Model(inputs, output)
        logits = m.predict(element['image'])
        output = {
            'logits': logits,
            'filename': element['filename'],
        }
        yield output


def build_pipeline(p, args):
    path = os.path.join(args.input_dir, "*", "*", "*")
    files = tf.io.gfile.glob(path)
    filenames = (
        p
        | "CreateFilePattern" >> beam.Create(files)
        # TODO: compare filenames' suffix to list of video suffix types
        | "FilterVideos" >> beam.Filter(lambda x: x.split(".")[-1] == "mkv")
        | "FilterVideos2" >> beam.Filter(lambda x: x.split("/")[-2] == "360P")
    )
    frames = (
        filenames
        | beam.ParDo(VideoToFrames(args.service_account_key_file))
        | beam.ParDo(Inception())
    )
    frames | beam.Map(print)
