from google.cloud import storage
import pyedflib

def list_bucket_directories(bucket_name):
    """Lists all directories in a Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    
    # List all blobs (files and directories) in the bucket
    blobs = bucket.list_blobs()

    # Collect directory names
    directories = set()
    for blob in blobs:
        # Blob name is in format: directory_name/file_name
        directory_name = blob.name.split('/')[0]
        directories.add(directory_name)
    
    return directories

def read_edf_from_gcp(bucket_name, file_name):
    """Reads an EDF file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    
    # Read the EDF file
    f = pyedflib.EdfReader('/tmp/temp.edf')
    print(f"Number of signals in the file: {f.signals_in_file}")


def list_buckets():
    """Lists all buckets."""

    storage_client = storage.Client()
    buckets = storage_client.list_buckets()

    for bucket in buckets:
        print(bucket.name)


if __name__ == "__main__":
    bucket_name = 'copenhagen_medtech_hackathon'
    file_name = 'copenhagen_medtech_hackathon/BrainCapture Dataset/v4.0/S001/S001R01.edf'
    list_bucket_directories(bucket_name)
