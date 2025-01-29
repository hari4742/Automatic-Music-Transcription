import os
import requests
import pandas as pd

# Metadata CSV URL for MAESTRO dataset
metadata_url = "https://huggingface.co/datasets/ddPn08/maestro-v3.0.0/resolve/main/maestro-v3.0.0.csv"
dataset_url_base = "https://huggingface.co/datasets/ddPn08/maestro-v3.0.0/resolve/main/"
dataset_path = "data/raw/maestro-v3.0.0"  # Directory where files will be saved

# Create the dataset directory if it doesn't exist
os.makedirs(dataset_path, exist_ok=True)

# Download metadata CSV
print("Downloading metadata CSV...")
response = requests.get(metadata_url)

# Save metadata CSV locally
metadata_path = os.path.join(dataset_path, "maestro_metadata.csv")
with open(metadata_path, "wb") as f:
    f.write(response.content)

print("Metadata CSV downloaded successfully!")

# Load the metadata CSV using pandas
metadata = pd.read_csv(metadata_path)

# Display first few rows to understand structure (optional)
print(metadata.head())

# Loop through the metadata to download files and recreate folder structure
for _, row in metadata.iterrows():
    midi_filename = row["midi_filename"]
    audio_filename = row["audio_filename"]

    # Construct the full path for MIDI and audio files based on the folder structure
    midi_url = dataset_url_base + midi_filename
    audio_url = dataset_url_base + audio_filename

    # Create directories for MIDI and audio files
    midi_path = os.path.join(dataset_path, os.path.dirname(midi_filename))
    audio_path = os.path.join(dataset_path, os.path.dirname(audio_filename))

    os.makedirs(midi_path, exist_ok=True)
    os.makedirs(audio_path, exist_ok=True)

    # Download MIDI file
    print(f"Downloading MIDI: {midi_filename}...")
    midi_response = requests.get(midi_url)
    with open(os.path.join(midi_path, os.path.basename(midi_filename)), "wb") as midi_file:
        midi_file.write(midi_response.content)

    # Download Audio file
    print(f"Downloading Audio: {audio_filename}...")
    audio_response = requests.get(audio_url)
    with open(os.path.join(audio_path, os.path.basename(audio_filename)), "wb") as audio_file:
        audio_file.write(audio_response.content)

print("Download completed for all selected files!")
