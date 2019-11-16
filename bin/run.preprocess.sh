#!/bin/bash

# Convenience script for running preproccessing pipeline.
#
# Arguments:
#   TYPE (optional): run type. If "cloud", then preprocessing will be ruun on
#       Dataflow.


. ./bin/_common.sh

TYPE=${1:-"local"}

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
        --cloud

    rm -rf *.egg-info

else
    python -m preprocessing.run_preprocess \
        --output_dir "${OUTPUT_DIR}" \
        --project_id "${PROJECT_ID}" \
        --input_dir "${INPUT_DIR}"
fi