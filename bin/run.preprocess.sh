#!/bin/bash

# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Convenience script for running preproccessing pipeline.
#
# Arguments:
#   SERVICE ACCCOUNT KEY: Path to service account key
#   MODE (optional): Mode indicating how many frames should be written to each sample.
#   TYPE (optional): run type. If "cloud", then preprocessing will be run on
#       Dataflow. If "local_crop", then preprocessing will be run locally with
#       cropping.

. ./bin/_common.sh

SA_KEY=${1:-GOOGLE_APPLICATION_CREDENTIALS}
MODE=${2:-"single_frame"}
TYPE=${3:-"local"}

PROJECT_ID="$(get_project_id)"
NOW="$(get_date_time)"
JOB_NAME="videos-to-tfrecords-${NOW}"
BUCKET="gs://${PROJECT_ID}/videos-to-tfrecords"
JOB_DIR="${BUCKET}/jobs/${JOB_NAME}"
INPUT_DIR="${BUCKET}/input"
OUTPUT_DIR="${BUCKET}/data/${NOW}"

if [ "${TYPE}" == "cloud" ]; then
    python -m preprocessing.run_preprocess \
        --job_name "${JOB_NAME}" \
        --job_dir "${JOB_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --project_id "${PROJECT_ID}" \
        --input_dir "${INPUT_DIR}" \
        --setup_file ./setup.py \
        --service_account_key_file "${SA_KEY}" \
        --mode "${MODE}" \
        --cloud
    rm -rf *.egg-info

else
    INPUT_DIR="gs://internal-klm/videos-to-tfrecords/input/Animation/360P"
    python -m preprocessing.run_preprocess \
        --output_dir "${OUTPUT_DIR}" \
        --project_id "${PROJECT_ID}" \
        --input_dir "${INPUT_DIR}" \
        --service_account_key_file "${SA_KEY}" \
        --mode "${MODE}" \

fi
