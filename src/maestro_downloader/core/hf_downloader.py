import logging
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from src.maestro_downloader.utils import ensure_directory_exists

logger = logging.getLogger(__name__)


class HuggingFaceDownloader:
    def __init__(self, repo_id: str, max_size: int, output_dir: str = "data/raw", metadata_dir: str = "data/metadata", metadata_filename: str = "metadata.csv"):
        self.repo_id = repo_id
        self.output_dir = ensure_directory_exists(output_dir)
        self.csv_path = Path(self.output_dir) / "maestro-v3.0.0.csv"
        self.metadata_dir = ensure_directory_exists(metadata_dir)
        self.metadata_file = self.metadata_dir / metadata_filename
        self.downloaded_files = []
        self.max_size: int = max_size

    def download_csv(self) -> None:
        """Download the CSV file from Hugging Face."""
        if not self.csv_path.exists():
            logger.info("Downloading dataset metadata...")
            hf_hub_download(
                repo_id=self.repo_id,
                filename="maestro-v3.0.0.csv",
                local_dir=self.output_dir,
                local_dir_use_symlinks=False,
                repo_type="dataset",
            )
            logger.info(f"Metadata downloaded to {self.csv_path}")

    def get_file_list(self) -> pd.DataFrame:
        """Get a list of audio and MIDI files with their sizes."""
        self.download_csv()
        df = pd.read_csv(self.csv_path)
        return df

    def download_subset(self) -> None:
        """Download files until the total size reaches the specified limit."""
        files = self.get_file_list()
        total_size = 0
        selected_files = []

        # Sort files by duration (shortest first)
        files = files.sort_values(by="duration", ascending=True)

        for _, row in files.iterrows():
            duration = row['duration']
            # Estimate file size based on duration (approximate)
            sample_rate = 48000
            bit_depth = 16
            channels = 2  # stereo
            audio_size = duration * sample_rate * \
                channels * bit_depth / 8  # 44.1–48 kHz 16-bit PCM stereo
            midi_size = duration * 1000  # Approximate MIDI size
            total_file_size = audio_size + midi_size

            if total_size + total_file_size > self.max_size:
                break
            selected_files.append(row)
            total_size += total_file_size

        logger.info(
            f"Selected {len(selected_files)} pairs totaling {total_size / (1024 ** 2):.2f} MB")

        selected_files = pd.DataFrame(selected_files)

        # Download the selected files
        for _, row in tqdm(selected_files.iterrows(), desc="Downloading files"):
            try:
                audio_file, midi_file = row['audio_filename'], row['midi_filename']
                # Download audio file
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=audio_file,
                    local_dir=self.output_dir,
                    local_dir_use_symlinks=False,
                    repo_type="dataset",
                )
                # Download MIDI file
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=midi_file,
                    local_dir=self.output_dir,
                    local_dir_use_symlinks=False,
                    repo_type="dataset",
                )

                self.downloaded_files.append(row)

                logger.info(
                    f"Download complete. Files saved to {self.output_dir}")
            except Exception as e:
                logger.error(
                    f"Error while downloading file from hugging face: {e}\nDownloaded only {len(self.downloaded_files)} to {self.output_dir}")

    def move_metadata_file(self):

        if len(self.downloaded_files) > 0:

            df = pd.DataFrame(self.downloaded_files)
            df.to_csv(self.metadata_file, index=False)

            logger.info(
                f"Successfully saved metadata file to {self.metadata_file}")
        else:
            logger.warning(
                f"No files downloaded yet. Download the files to save the metadata.")

    def run(self):
        self.download_subset()
        self.move_metadata_file()
