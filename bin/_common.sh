#!/bin/bash

#Common code used in shell scripts.

function get_project_id {
    echo "$(gcloud config list --format 'value(core.project)' 2>/dev/null)" 
}

function get_date_time {
  echo "$(date +%Y%m%d%H%M%S)"
}
