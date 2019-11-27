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


def generate_download_signed_url_v4(service_account_file, bucket_name, blob_name):
    """Generates a v4 signed URL for downloading a blob."""
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


class ConcatPaths(beam.DoFn):
    """Transform to create file paths."""
    def process(self, element):
        """Concatenates directory and filename."""
        for file in element[2]:
            logging.info(os.path.join(element[0], file))
            yield os.path.join(element[0], file)


class VideoToFrames(beam.DoFn):
    def __init__(self, service_account_file):
        self.service_account_file = service_account_file

    def process(self, filename):
        u = urllib.parse.urlparse(filename)
        signed_url = generate_download_signed_url_v4(
            self.service_account_file, u.netloc, u.path[1:])
        input_video = cv2.VideoCapture(signed_url)
        result, image = input_video.read()
        input_video.release()
        logging.info(image)
        return [image]


def build_pipeline(p, args):
    filenames = (
        p
        | "CreateFilePattern" >> beam.Create([args.input_dir])
        | "GetWalks" >> beam.ParDo(GetFilenames())
        # TODO: compare filenames' suffix to list of video suffix types
        | "FilterVideos" >> beam.Filter(lambda x: x.split(".")[-1] == "mkv")
    )
    key_file = "gs://internal-klm/videos-to-tfrecords/internal-klm-c5ab3b3ba702.json"
    frames = (
        filenames
        | beam.ParDo(VideoToFrames(key_file))
        | beam.Map(lambda x: (1, x))
    )
    filenames | beam.Map(print)

