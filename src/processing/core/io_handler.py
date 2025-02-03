import h5py
from pathlib import Path
import numpy as np


class HDF5Writer:
    def __init__(self, output_path: str):
        """
        Initialize the HDF5 writer.

        Args:
            output_path (str): Path to the output HDF5 file.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_data(self, split: str, year: str, filename: str, cqt: np.ndarray, pianoroll: np.ndarray):
        """
        Write processed data to the HDF5 file.

        Args:
            split (str): Dataset split (train/validation/test).
            year (str): Year of the recording.
            filename (str): Name of the file.
            cqt (np.ndarray): CQT spectrogram.
            pianoroll (np.ndarray): Piano roll.
        """
        with h5py.File(self.output_path, "a") as hdf:
            group_path = f"{split}/{year}/{filename}"
            group = hdf.require_group(group_path)
            group.create_dataset("cqt", data=cqt, compression="gzip")
            group.create_dataset(
                "pianoroll", data=pianoroll, compression="gzip")
