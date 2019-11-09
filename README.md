# videos-to-tfrecords
Library to convert video files to TFRecords using Apache Beam.

## Set up
### Set up GCP credentials
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project {project-id}
```

### Set up Python environment
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### Prepare video files (input data)
The Apache Beam pipeline will use Google Cloud Storage as the source.

Both the README and in the Bash scripts assume that you have created a GCS Bucket name gs://{project-id}. If you have not created this bucket yet, create it using the following `gsutil` command:

```bash
gsutil mb gs://{project-id}
```

```bash
gsutil -m cp -r gs://ugc-dataset/original_videos/ gs://{project-id}/videos-to-tfrecords/input/
```

## Run Script
### Run locally
Useful for testing and debugging
```bash
bash bin/run.preprocess.sh
```

### Run on Cloud Dataflow
```bash
bash bin/run.preprocess.sh cloud
```
