import os
from pathlib import Path
import requests
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def ensure_directory_exists(path: str) -> Path:
    """Ensure the directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file_with_progress(
    url: str, destination: str, chunk_size: int = 8192
) -> None:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(
            total=total_size, unit="B", unit_scale=True, desc=f"Downloading {Path(destination).name}"
        )

        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                progress_bar.update(len(chunk))
        progress_bar.close()
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def delete_file(file_path: str) -> None:
    """Delete a file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
