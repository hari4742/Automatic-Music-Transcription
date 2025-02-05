import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
from src.maestro_downloader.core.google_downloader import GoogleStorageDownloader
from src.maestro_downloader.core.hf_downloader import HuggingFaceDownloader
from src.utils.logger import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="maestro_downloader_config")
def main(cfg: DictConfig) -> None:
    """Main function to handle dataset downloads."""
    setup_logging(cfg.logs.output_dir, cfg.logs.output_filename)
    logger = logging.getLogger(__name__)
    logger.info("Starting Maestro Dataset Downloader...")

    try:
        # Ask user for download source
        source = input(
            "Choose download source:\n"
            "[1] Google Storage (full dataset)\n"
            "[2] Hugging Face (subset)\n"
            "Enter choice (1 or 2): "
        ).strip()

        if source == "1":
            # Initialize Google Storage Downloader
            logger.info("Downloading full dataset from Google Storage...")
            google_downloader = GoogleStorageDownloader(
                download_url=cfg.datasets.google.url,
                output_dir=cfg.datasets.google.output_dir,
                metadata_dir=cfg.datasets.google.metadata_dir,
                metadata_filename=cfg.datasets.google.metadata_filename
            )
            google_downloader.run()
            logger.info(
                "Full dataset download and extraction completed successfully.")

        elif source == "2":
            # Initialize Hugging Face Downloader
            try:
                max_size = int(
                    input("Enter maximum size to download (in MB, e.g., 500): ").strip()
                )
                if max_size <= 0:
                    raise ValueError("Size must be a positive number.")
            except ValueError as e:
                logger.error(f"Invalid input: {e}")
                return

            logger.info(
                f"Downloading subset (max {max_size} MB) from Hugging Face...")
            hf_downloader = HuggingFaceDownloader(
                repo_id=cfg.datasets.hf.repo_id,
                max_size=max_size * 1024 * 1024,  # Convert MB to bytes
                output_dir=cfg.datasets.hf.output_dir,
            )
            hf_downloader.run()
            logger.info("Subset download completed successfully.")

        else:
            logger.error("Invalid choice. Please enter 1 or 2.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        logger.info("Please check the logs and try again.")


if __name__ == "__main__":
    main()
