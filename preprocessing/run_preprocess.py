"""Preprocess video files into TFRecords using Apache Beam."""

import argparse
import logging
from datetime import datetime
import apache_beam as beam
import sys

def parse_arguments(argv):
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Runs preprocessing for video data to convert to TFRecords.")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    parser.add_argument(
        "--cloud",
        nags="?",
        const=True,
        default=False,
        type=bool,
        help="""If specified, runs preprocessing on GCP with the Dataflow
        Runner. Otherwise, preprocessing is run locally.""")
    parser.add_argument(
        "--job_name",
        help="Dataflow job name.",
        type=str,
        deafult="{}-{}".format("video-to-tfrecords", timestamp))
    parser.add_argument(
        "--job_dir",
        type=str,
        help="""GCS bucket to stage code and write temporaryy outputs for cloud
        runs.""")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="""Local directory or GCS bucket to write training, validation,
        and testing TFRecords.""")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="""Local directory or GCS bucket containing video files.""")
    parser.add_argument(
        "--log_level",
        default="INFO",
        type=str,
        help="Set logging level.")
    parser.add_argument(
        "--project_id",
        help="GCP project id",
        type=str,
        required=True)
    args, _ = parser.parse_known_args(args=argv[1:])
    return args


def main():
    """Configures and runs an Apache Beam pipeline."""
    args = parse_arguments(sys.args)
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))


if __name__ == "__main__":
    main()
