"""Build preprocessing pipeline."""

import apache_beam as beam
import tensorflow as tf
import os


class GetFilenames(beam.DoFn):
    def process(self, path):
        """Use tf.io.gfile.walk to support recursive lookups."""
        return tf.io.gfile.walk(path)


class ConcatPaths(beam.DoFn):
    def process(self, element):
        for file in element[2]:
            yield os.path.join(element[0], file)


def build_pipeline(p, args):
    data = (
        p
        | "CreateFilePattern" >> beam.Create([args.input_dir])
        | "GetWalks" >> beam.ParDo(GetFilenames())
        | "FilterSubdirectories" >> beam.Filter(lambda x: not len(x[1]))
        | "ConcatPaths" >> beam.ParDo(ConcatPaths())
    )
    data | beam.Map(print)
