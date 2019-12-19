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
"""Preprocess video files into TFRecords using Apache Beam."""

import argparse
import logging
from datetime import datetime
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, WorkerOptions

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
        "--machine_type",
        help="""Set machine type for Dataflow worker machines.""",
        default="n1-highcpu-16")
    parser.add_argument(
        "--batch_size",
        help="Batch size for TensorFlow Hub predictions.",
        default=1)
    parser.add_argument(
        "--output_dir",
        type=str,
        help="""Local directory or GCS bucket to write training, validation,
        and testing TFRecords.""",
        required=True)
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
    parser.add_argument(
        "--mode",
        choices=["crop_video", "single_frame", "full_video"],
        help="""Indicator for how many frames should be stored for each
        sample.""",
        default="single_frame")
    parser.add_argument(
        "--sequence_length",
        type=int,
        help="""Length (in milliseconds) to crop each video sample. This corresponds
        to window length.""",
        default=15000)
    parser.add_argument(
        "--period",
        type=int,
        help="""Sliding window period (in milliseconds) between samples. If
        left empty, fixed windows are used.""",
        default=1000)
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
        pipeline_options = PipelineOptions(flags=[], **options)
        pipeline_options.view_as(WorkerOptions).machine_type = args.machine_type
    else:
        pipeline_options = PipelineOptions(flags=[], **options)
    return pipeline_options


def main():
    """Configures and runs an Apache Beam pipeline."""
    args = parse_arguments(sys.argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    logging.info("mode: " + args.mode)
    options = get_pipeline_options(args)
    runner = "DataflowRunner" if args.cloud else "DirectRunner"
    with beam.Pipeline(runner, options=options) as pipeline:
        preprocess.build_pipeline(pipeline, args)


if __name__ == "__main__":
    main()
