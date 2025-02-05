import logging
import zipfile
from tqdm import tqdm
import shutil
from src.maestro_downloader.utils import (
    ensure_directory_exists,
    download_file_with_progress,
    delete_file,
)

logger = logging.getLogger(__name__)


class GoogleStorageDownloader:
    def __init__(self, download_url: str, output_dir: str = "data/raw", metadata_dir: str = "data/metadata", metadata_filename: str = "metadata.csv"):
        self.download_url = download_url
        self.output_dir = ensure_directory_exists(output_dir)
        self.zip_path = self.output_dir / "maestro-v3.0.0.zip"
        self.metadata_dir = ensure_directory_exists(metadata_dir)
        self.metadata_file = self.metadata_dir / metadata_filename

    def download(self) -> None:
        """Download the dataset from Google Storage."""
        logger.info(f"Downloading dataset from {self.download_url}...")
        download_file_with_progress(self.download_url, self.zip_path)

    def extract(self) -> None:
        """Extract the downloaded ZIP file."""
        logger.info(f"Extracting {self.zip_path}...")
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc="Extracting"):
                zip_ref.extract(file, self.output_dir)
        logger.info(f"Extraction complete. Files saved to {self.output_dir}")

    def cleanup(self) -> None:
        """Delete the ZIP file after extraction."""
        delete_file(self.zip_path)

    def move_metadata_file(self):

        try:

            dest_metadata_file = self.metadata_file
            source_metadata_file = self.output_dir / "maestro-v3.0.0.csv"

            shutil.copy(source_metadata_file, dest_metadata_file)

            logger.info(
                f"Successfully moved metadata file to {dest_metadata_file}")
        except Exception as e:
            logger.error(
                f"Error in moving the metadata file to metadata folder: {e}")

    def run(self) -> None:
        """Run the full download and extraction process."""
        self.download()
        self.extract()
        self.cleanup()
        self.move_metadata_file()
