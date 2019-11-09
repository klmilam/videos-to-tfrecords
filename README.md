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

Preprocessing scripts assume that you've created a GCS bucket gs://{project-id}