import logging
import zipfile
from tqdm import tqdm
from src.maestro_downloader.utils import (
    ensure_directory_exists,
    download_file_with_progress,
    delete_file,
)

logger = logging.getLogger(__name__)


class GoogleStorageDownloader:
    def __init__(self, download_url: str, output_dir: str = "data/raw"):
        self.download_url = download_url
        self.output_dir = ensure_directory_exists(output_dir)
        self.zip_path = self.output_dir / "maestro-v3.0.0.zip"

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

    def run(self) -> None:
        """Run the full download and extraction process."""
        self.download()
        self.extract()
        self.cleanup()
        # TODO: move csv file to metadata folder
