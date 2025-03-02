import torch
import torch.nn as nn


class MultiPitchEstimator(nn.Module):
    def __init__(self, input_bins: int):
        """
        Initialize the multi-pitch estimation model.

        Args:
            input_bins (int): Number of frequency bins in the CQT spectrogram.
        """
        super().__init__()

        # CNN Block
        self.conv = nn.Conv2d(
            in_channels=1,  # Single channel (mono audio)
            out_channels=64,
            kernel_size=(20, 2),  # Time x Frequency
            stride=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2))
        self.dropout = nn.Dropout(0.3)

        # Calculate CNN output dimensions
        cnn_time_dim = (360 - 20) // 4 + 1  # Adjusted for pooling
        cnn_freq_dim = (input_bins - 2) // 2 + 1
        lstm_input_size = cnn_freq_dim * 64  # Flattened CNN output

        # RNN Block
        self.lstm1 = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=500,
            bidirectional=True,
            # dropout=0.75  # Applied only to first LSTM
        )
        self.lstm2 = nn.LSTM(
            input_size=1000,  # Bidirectional output
            hidden_size=200,
            bidirectional=True
        )

        # Output Layer
        self.fc = nn.Linear(400, 88)  # 200 * 2 (bidirectional)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input CQT spectrogram of shape (batch, sequence, freq).

        Returns:
            torch.Tensor: Predicted piano roll of shape (batch, sequence, 88).
        """

        print(f"Input Shape: {x.shape}")
        # Input shape: (batch, sequence, freq) → (batch, 1, sequence, freq)
        x = x.unsqueeze(1)

        # CNN Block
        x = self.pool(torch.relu(self.conv(x)))
        x = self.dropout(x)

        # Flatten frequency and channels
        batch, channels, time, freq = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, time, -1)

        # RNN Block
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Output Layer
        return torch.sigmoid(self.fc(x))
