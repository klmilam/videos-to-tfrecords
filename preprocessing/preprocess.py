"""Build preprocessing pipeline."""

import apache_beam as beam
import tensorflow as tf
import os
import numpy as np
import io
import os
import tempfile
import subprocess


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
        return tf.io.gfile.walk(path)


class ConcatPaths(beam.DoFn):
    """Transform to create file paths."""
    def process(self, element):
        """Concatenates directory and filename."""
        for file in element[2]:
            yield os.path.join(element[0], file)


class VideoToFrames(beam.DoFn):
    def process(self, filename):
        local_input_file = tempfile.NamedTemporaryFile(suffix=".mkv").name
        tf.io.gfile.copy(filename, local_input_file)
        command = ["ffmpeg",
            "-y",
            "-i", local_input_file,
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-codec:v", "mpeg4", "-"]
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
        raw_image = pipe.stdout.read(420*360*3)
        image =  np.fromstring(raw_image, dtype='uint8')
        pipe.stdout.flush()
        return [image]


def build_pipeline(p, args):
    filenames = (
        p
        | "CreateFilePattern" >> beam.Create([args.input_dir])
        | "GetWalks" >> beam.ParDo(GetFilenames())
        | "ConcatPaths" >> beam.ParDo(ConcatPaths())
        # TODO: compare filenames' suffix to list of video suffix types
        | "FilterVideos" >> beam.Filter(lambda x: x.split(".")[-1] == "mkv")
    )
    frames = (
        filenames | beam.ParDo(VideoToFrames())
    )
    filenames | beam.Map(print)

