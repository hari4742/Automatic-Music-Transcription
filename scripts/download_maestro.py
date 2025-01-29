'''
This file downloads the MAESTRO dataset from the Google Cloud Storage bucket and extracts it to the data/raw directory.

It requires 120 GB of free disk space to download and extract the dataset.
'''

import os
import requests
import zipfile

# URL for MAESTRO dataset
url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
dataset_path = "data/raw"
zip_path = "data/maestro-v3.0.0.zip"

# Create directories if they don't exist
os.makedirs(dataset_path, exist_ok=True)

# Download dataset
print("Downloading MAESTRO dataset... This may take some time.")
response = requests.get(url, stream=True)

with open(zip_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        file.write(chunk)

print("Download complete. Extracting...")

# Extract dataset
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(dataset_path)

# Remove the zip file
os.remove(zip_path)

print("Dataset extracted successfully!")
