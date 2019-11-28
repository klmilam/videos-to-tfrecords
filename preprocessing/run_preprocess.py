"""Preprocess video files into TFRecords using Apache Beam."""

import argparse
import logging
from datetime import datetime
import apache_beam as beam
import sys
import os

from preprocessing import preprocess


def parse_arguments(argv):
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Runs preprocessing for video data to convert to TFRecords.")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    parser.add_argument(
        "--cloud",
        help="""Run preprocessing on the cloud. Default False.""",
        action='store_true',
        default=False)
    parser.add_argument(
        "--job_name",
        help="Dataflow job name.",
        type=str,
        default="{}-{}".format("video-to-tfrecords", timestamp))
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
    parser.add_argument(
        "--setup_file",
        default="./setup.py",
        type=str,
        help="""Path to setup.py file.""")
    parser.add_argument(
        "--service_account_key_file",
        type=str,
        help="""Path to service account key file. If the job is running on the
        cloud, the file should be stored on GCS.""",
        required=True)
    parser.add_argument(
        "--frame_sample_rate",
        type=int,
        help="Number of milliseconds between each sample. Default is 500.",
        default=500)
    args, _ = parser.parse_known_args(args=argv[1:])
    return args


def get_pipeline_options(args):
    """Returns pipeline options."""
    options = {"project": args.project_id}
    if args.cloud:
        if not args.job_dir:
            raise ValueError("Job directory must be specified for Dataflow.")
        if args.service_account_key_file.split(":")[0] != "gs":
            raise ValueError("""Service account keys must be uploaded to GCS
                for Dataflow.""")
        options.update({
            "job_name": args.job_name,
            "setup_file": args.setup_file,
            "staging_location": os.path.join(args.job_dir, "staging"),
            "temp_location": os.path.join(args.job_dir, "tmp"),
        })
    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **options)
    return pipeline_options


def main():
    """Configures and runs an Apache Beam pipeline."""
    args = parse_arguments(sys.argv[1:])
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    options = get_pipeline_options(args)
    runner = "DataflowRunner" if args.cloud else "DirectRunner"
    with beam.Pipeline(runner, options=options) as pipeline:
        preprocess.build_pipeline(pipeline, args)


if __name__ == "__main__":
    main()
