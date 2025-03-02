import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional


class MaestroDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        split: str,
        sequence_length: int = 360,
        step_size: int = 720,
        context_frames: int = 2,
        max_length: Optional[int] = None
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.split = split
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.context_frames = context_frames
        self.max_length = max_length

        # Collect all valid sequences
        self.sequences = []
        with h5py.File(hdf5_path, 'r') as hdf:
            if split not in hdf:
                raise ValueError(f"Split {split} not found in HDF5 file")

            for year in hdf[split]:
                for piece in hdf[f"{split}/{year}"]:
                    group = hdf[f"{split}/{year}/{piece}"]
                    cqt = group['cqt']
                    total_frames = cqt.shape[0]

                    # Calculate valid sequence start indices
                    starts = range(
                        0,
                        total_frames - sequence_length + 1,
                        step_size
                    )

                    if self.max_length:
                        starts = starts[:self.max_length]

                    for start in starts:
                        self.sequences.append((
                            f"{split}/{year}/{piece}",
                            start,
                            start + sequence_length
                        ))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        group_path, start, end = self.sequences[idx]

        with h5py.File(self.hdf5_path, 'r') as hdf:
            group = hdf[group_path]
            cqt = group['cqt'][start:end]  # (seq_len, 288)
            pianoroll = group['pianoroll'][start:end]  # (seq_len, 88)

            # Add temporal context to CQT
            cqt_context = self._add_context(cqt)
            # Convert to tensors
            cqt_tensor = torch.from_numpy(
                cqt_context.copy()).float().unsqueeze(1)  # (seq_len, 1, 288, 5)
            pianoroll_tensor = torch.from_numpy(pianoroll.copy()).float()

            return cqt_tensor, pianoroll_tensor

    def _add_context(self, cqt: np.ndarray) -> np.ndarray:
        """Add temporal context to CQT frames."""
        # Pad to handle edge frames
        padded = np.pad(
            cqt,
            [(self.context_frames, self.context_frames), (0, 0)],
            mode='reflect'
        )

        # Create sliding windows of shape (seq_len, context_frames, freq_bins)
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (2 * self.context_frames + 1, cqt.shape[1])
        )
        # Remove singleton dimension and transpose to (seq_len, freq_bins, context)
        windows = windows.squeeze(axis=1).transpose(0, 2, 1)

        return windows  # Shape: (seq_len, 288, 5)
