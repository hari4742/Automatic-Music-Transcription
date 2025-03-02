import torch
import torch.nn as nn


class MultiPitchEstimator(nn.Module):
    def __init__(self):
        """
        Initialize the multi-pitch estimation model.

        Args:
            input_bins (int): Number of frequency bins in the CQT spectrogram.
        """
        super().__init__()

        input_shape = (288, 5)
        kernel_size = (20, 2)
        out_channels = 32
        max_pool_kernel_shape = (4, 2)

        # CNN for spectrogram feature extraction
        self.cnn = nn.Sequential(
            # Input: (1, 288, 5) Output: (32, 269, 4)
            nn.Conv2d(1, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            # Output: (32, 67, 2)
            nn.MaxPool2d(max_pool_kernel_shape),
            nn.Flatten(),                           # Output: 32*67*2 = 4288
        )

        x = ((input_shape[0]-kernel_size[0])+1)//max_pool_kernel_shape[0]
        y = ((input_shape[1]-kernel_size[1])+1)//max_pool_kernel_shape[1]
        lstm_input_size = out_channels * x * y

        # Bidirectional LSTMs
        self.lstm1 = nn.LSTM(input_size=lstm_input_size, hidden_size=500,
                             bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.75)

        self.lstm2 = nn.LSTM(input_size=1000, hidden_size=200,  # 500*2 bidirectional
                             bidirectional=True, batch_first=True)

        # self.lstm = nn.Sequential(
        #     nn.LSTM(input_size=lstm_input_size, hidden_size=500,
        #             bidirectional=True, batch_first=True),
        #     nn.Dropout(0.75),  # Recurrent dropout between LSTM layers
        #     nn.LSTM(input_size=1000, hidden_size=200,  # 500*2 bidirectional
        #             bidirectional=True, batch_first=True)
        # )

        # Final output layer
        self.fc = nn.Sequential(
            nn.Linear(400, 88),  # 200*2 bidirectional
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input shape: (batch_size, seq_len, 1, freq_bins, time_steps)
        batch_size, seq_len = x.size(0), x.size(1)

        # Process each timestep through CNN
        x = x.view(batch_size * seq_len, 1, 288, 5)
        x = self.cnn(x)  # (batch_size*seq_len, 4288)
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 4288)

        # Process through bidirectional LSTMs
        x, _ = self.lstm1(x)  # LSTM returns (output, (h_n, c_n))
        x = self.dropout(x)   # Apply dropout to LSTM output
        x, _ = self.lstm2(x)  # Pass through second LSTM

        # Final output layer
        return self.fc(x)  # (batch_size, seq_len, 88)
