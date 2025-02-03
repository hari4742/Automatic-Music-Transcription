from pathlib import Path
import logging


def setup_logging(output_dir: str, filename: str) -> None:
    """Configure logging to write to both console and file."""
    # Ensure the output directory exists
    log_file = Path(output_dir) / filename
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Clear existing handlers (if any)
    logging.getLogger().handlers.clear()

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    # Set log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler],
    )

    logging.info(f"Logging initialized. Logs will be saved to {log_file}")
