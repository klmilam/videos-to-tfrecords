# videos-to-tfrecords
Library to convert video files to TFRecords using Apache Beam.

## Set up
### Set up GCP credentials
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project {project-id}
```
If you will be running the pipeline on the Dataflow Runner, the service account key should be accessible to the Dataflow workers. So, the file should be copied from its local path to Google Cloud Storage.
```bash
gsutil cp {local-path-to-json} {cloud-storage-path-to-json}
```
### Set up Python environment
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### Set up environment variables
```
PROJECT_ID={project-id}
```

### Prepare video files (input data)
The Apache Beam pipeline will use Google Cloud Storage as the source.

Both the README and in the Bash scripts assume that you have created a GCS Bucket name gs://{project-id}. If you have not created this bucket yet, create it using the following `gsutil` command:

```bash
gsutil mb gs://{project-id}
```

```bash
gsutil -m cp -r gs://ugc-dataset/original_videos/* \
    gs://${PROJECT_ID}/videos-to-tfrecords/input/
```


## Run Script
The Bash scripts below assume that you have created a GCS directory gs://{project-id}/videos-to-tfrecords/input/ which stores your training data.
#TODO(kmilam): Support custom input directories

### Run locally
Running an Apache Beam pipeline locally can be helpful for testing and debugging. However, it's not recommended when working with a lot of data; use the Cloud Dataflow runner for Apache Beam instead.
```bash
bash bin/run.preprocess.sh {cloud-storage-path-to-json}
```

### Run on Cloud Dataflow
```bash
bash bin/run.preprocess.sh {cloud-storage-path-to-json} cloud
```
