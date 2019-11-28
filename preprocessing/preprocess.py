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
    os.remove(local_key)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version='v4',
        expiration=datetime.timedelta(minutes=15),
        method='GET')
    return url


class GetFilenames(beam.DoFn):
    """Transform to list contents of directory recursively."""
    def process(self, path):
        """Returns contents of every directory."""
        path = os.path.join(path, "*", "*", "*")
        return tf.io.gfile.glob(path)


class VideoToFrames(beam.DoFn):
    """Transform to read a video file from GCS and extract frames."""
    def __init__(self, service_account_file, skip_msec):
        self.service_account_file = service_account_file
        self.skip_msec = skip_msec

    def process(self, filename):
        u = urllib.parse.urlparse(filename)
        signed_url = generate_download_signed_url_v4(
            self.service_account_file, u.netloc, u.path[1:])
        video = cv2.VideoCapture(signed_url)

        last_ts = -9999
        result, image = video.read()
        while(video.isOpened()):
            # Only record frames occurring every skip_msec
            while video.get(cv2.CAP_PROP_POS_MSEC) < self.skip_msec + last_ts:
                result, image = video.read()
                if not result:
                    return
            last_ts = video.get(cv2.CAP_PROP_POS_MSEC)
            image = image/255.  # Normalize
            image = image[:, :, ::-1]  # OpenCV orders channels BGR
            image = image[np.newaxis, :, :, :]  # Add batch dimension
            output = {
                'image': image,
                'filename': filename,
                'timestamp_ms': last_ts,
                'frame_per_sec': round(video.get(cv2.CAP_PROP_FPS)),
                'frame_total': video.get(cv2.CAP_PROP_FRAME_COUNT),
            }
            yield output
        video.release()



class Inception(beam.DoFn):
    """Transform to extract Inception-V3 bottleneck features."""
    def process(self, element):
        inputs = tf.keras.Input(shape=(None, None, 3))
        inception_layer = hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
            output_shape=2048,
            trainable=False
        )
        output = inception_layer(inputs)
        model = tf.keras.Model(inputs, output)
        logits = model.predict(element['image'])
        del element['image']
        element['logits'] = logits
        yield element


def build_pipeline(p, args):
    path = os.path.join(args.input_dir, "*", "*", "*")
    files = tf.io.gfile.glob(path)
    filenames = (
        p
        | "CreateFilePattern" >> beam.Create(files)
        # TODO: compare filenames' suffix to list of video suffix types
        | "FilterVideos" >> beam.Filter(lambda x: x.split(".")[-1] == "mkv")
    )
    frames = (
        filenames
        | beam.ParDo(VideoToFrames(
            args.service_account_key_file, args.frame_sample_rate))
        | beam.ParDo(Inception())
    )
    frames | beam.Map(print)
