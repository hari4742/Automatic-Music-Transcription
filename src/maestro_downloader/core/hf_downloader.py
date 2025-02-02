import logging
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from src.maestro_downloader.utils import ensure_directory_exists

logger = logging.getLogger(__name__)


class HuggingFaceDownloader:
    def __init__(self, repo_id: str, output_dir: str = "data/raw"):
        self.repo_id = repo_id
        self.output_dir = ensure_directory_exists(output_dir)
        self.csv_path = Path(self.output_dir) / "maestro-v3.0.0.csv"

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
            logger.info(f"Metadata saved to {self.csv_path}")

    def get_file_list(self) -> List[Tuple[str, str, float]]:
        """Get a list of audio and MIDI files with their sizes."""
        self.download_csv()
        df = pd.read_csv(self.csv_path)
        file_list = []
        for _, row in df.iterrows():
            audio_file = row["audio_filename"]
            midi_file = row["midi_filename"]
            duration = row["duration"]
            file_list.append((audio_file, midi_file, duration))
        return file_list

    def download_subset(self, max_size: int) -> None:
        """Download files until the total size reaches the specified limit."""
        files = self.get_file_list()
        total_size = 0
        downloaded_files = []  # TODO: save the files which we downloaded into metadata folder

        # Sort files by duration (shortest first)
        files.sort(key=lambda x: x[2])

        for audio_file, midi_file, duration in files:
            # Estimate file size based on duration (approximate)
            sample_rate = 48000
            bit_depth = 16
            channels = 2  # stereo
            audio_size = duration * sample_rate * \
                channels * bit_depth / 8  # 44.1–48 kHz 16-bit PCM stereo
            midi_size = duration * 1000  # Approximate MIDI size
            total_file_size = audio_size + midi_size

            if total_size + total_file_size > max_size:
                break
            downloaded_files.append((audio_file, midi_file))
            total_size += total_file_size

        logger.info(
            f"Selected {len(downloaded_files)} pairs totaling {total_size / (1024 ** 2):.2f} MB")

        # Download the selected files
        for audio_file, midi_file in tqdm(downloaded_files, desc="Downloading files"):
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
                repo_type="dataset",  # Specify repo type as dataset
            )

        logger.info(f"Download complete. Files saved to {self.output_dir}")
