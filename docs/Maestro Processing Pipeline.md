# Maestro Dataset Processing Pipeline

A Python pipeline to process the MAESTRO dataset, converting WAV and MIDI files into spectrograms and piano rolls stored in HDF5 format.

## Features

- Convert WAV files to Constant-Q Transform (CQT) spectrograms.
- Convert MIDI files to piano rolls.
- Parallelized processing using multiprocessing.
- Save processed data in HDF5 format with train/validation/test splits.

## Usage

```bash
python src/processing/main.py
```

## Configuration

Edit `src/processing/config/config.yaml` to customize processing parameters.

## Output

- Processed data is saved in `data/processed/maestro.hdf5`

- Logs are saved to `src/logs/maestro_pipeline.log`.
