# Maestro Dataset Downloader

A Python CLI application to download the Maestro dataset from Google Storage or Hugging Face.

## Features

- Download the full dataset from Google Storage.
- Download a subset of the dataset from Hugging Face based on size.
- Progress tracking and logging.

### Usage

1. Run the application:

```bash
python src/maestro_downloader/main.py
```

2. Choose the download source:

- `1` for Google Storage (full dataset).

- `2` for Hugging Face (subset).

3. If choosing Hugging Face, enter the maximum size to download (in MB).

## Configuration

Edit [`src/configs/maestro_downloader_config.yaml`](../src/configs/maestro_downloader_config.yaml) to customize download settings.

## Logs

Logs are saved to `src/logs/maestro_downloader.log`.
