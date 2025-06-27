import papermill as pm
import datetime
import os
from google.cloud import storage

# Define parameters
bucket_name = "churn-model-prediction"
source_blob_name = "Telecom-Customer-Churn.csv"

# Use a writable local directory
os.makedirs("tmp", exist_ok=True)
destination_file_name = os.path.join("tmp", "Telecom-Customer-Churn.csv")

# Function to download blob from GCS
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

# Download the file
download_blob(bucket_name, source_blob_name, destination_file_name)

# Create timestamped output path
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"executed_notebooks/modelPrototype_1_output_{timestamp}.ipynb"

# Execute the notebook with parameters
pm.execute_notebook(
    input_path="modelPrototype.ipynb",
    output_path=output_path,
    parameters={
        "bucket_name": bucket_name,
        "source_blob_name": source_blob_name,
        "destination_file_name": destination_file_name
    },
    kernel_name="python3"
)
