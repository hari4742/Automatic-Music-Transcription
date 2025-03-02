import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MaestroDataset(Dataset):
    def __init__(self, hdf5_path: str, split: str, sequence_length: int = 360, stride: int = 720, freq_bins: int = 288):
        """
        Initialize the dataset.

        Args:
            hdf5_path (str): Path to the HDF5 file.
            split (str): Dataset split ("train", "validation", or "test").
            sequence_length (int): Length of each sequence (default: 360).
            stride (int): Stride for sequence sampling (default: 720).
        """
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.freq_bins = freq_bins

        # Load metadata and precompute sequence indices
        self.hdf5_file = h5py.File(self.hdf5_path, "r")
        self.split_group = self.hdf5_file[split]
        self.sequence_indices = self._precompute_sequence_indices()

    def _precompute_sequence_indices(self):
        """
        Precompute indices for all sequences in the dataset.

        Returns:
            list: List of tuples (group_name, start_idx).
        """
        sequence_indices = []
        for year in self.split_group:
            year_group = self.split_group[year]
            for filename in year_group:
                file_group = year_group[filename]
                cqt = file_group["cqt"][:]
                num_sequences = (
                    cqt.shape[0] - self.sequence_length) // self.stride + 1
                print(
                    f"File: {filename}, CQT shape: {cqt.shape}, Num sequences: {num_sequences}")
                for i in range(num_sequences):
                    start_idx = i * self.stride
                    sequence_indices.append((f"{year}/{filename}", start_idx))
        return sequence_indices

    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        """
        Get a sequence of CQT and piano roll.

        Args:
            idx (int): Index of the sequence.

        Returns:
            tuple: (cqt_sequence, pianoroll_sequence).
        """
        group_name, start_idx = self.sequence_indices[idx]
        file_group = self.split_group[group_name]

        # Load CQT and piano roll
        cqt = file_group["cqt"][start_idx:start_idx + self.sequence_length]
        pianoroll = file_group["pianoroll"][start_idx:start_idx +
                                            self.sequence_length]

        # Convert to PyTorch tensors
        cqt_tensor = torch.tensor(cqt, dtype=torch.float32)
        pianoroll_tensor = torch.tensor(pianoroll, dtype=torch.float32)

        return cqt_tensor, pianoroll_tensor

    def close(self):
        """Close the HDF5 file."""
        self.hdf5_file.close()
