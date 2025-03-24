import torch
import torch.nn as nn


class MultiPitchEstimator(nn.Module):
    def __init__(self):
        """
        Initialize the multi-pitch estimation model with additional CNN layer.
        """
        super().__init__()

        input_shape = (288, 5)
        # Parameters for the first CNN layer
        kernel_size1 = (10, 2)
        out_channels1 = 32
        max_pool_kernel1 = (4, 2)
        # Parameters for the second CNN layer
        kernel_size2 = (3, 2)
        out_channels2 = 64
        max_pool_kernel2 = (2, 1)

        # CNN for spectrogram feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, out_channels=out_channels1, kernel_size=kernel_size1),
            nn.ReLU(),
            nn.MaxPool2d(max_pool_kernel1),
            nn.Conv2d(out_channels1, out_channels=out_channels2,
                      kernel_size=kernel_size2),
            nn.ReLU(),
            nn.MaxPool2d(max_pool_kernel2),
            nn.Flatten(),
        )

        # Calculate LSTM input size after two CNN layers
        h, w = input_shape
        # First Conv and Pool
        h = (h - kernel_size1[0] + 1)  # Conv output height
        w = (w - kernel_size1[1] + 1)  # Conv output width
        h = h // max_pool_kernel1[0]    # MaxPool output height
        w = w // max_pool_kernel1[1]    # MaxPool output width
        # Second Conv and Pool
        h = (h - kernel_size2[0] + 1)   # Conv output height
        w = (w - kernel_size2[1] + 1)   # Conv output width
        h = h // max_pool_kernel2[0]    # MaxPool output height
        w = w // max_pool_kernel2[1]    # MaxPool output width
        lstm_input_size = out_channels2 * h * w

        # Bidirectional LSTMs
        self.lstm1 = nn.LSTM(input_size=lstm_input_size, hidden_size=500,
                             bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.75)

        self.lstm2 = nn.LSTM(input_size=1000, hidden_size=200,  # 500*2 bidirectional
                             bidirectional=True, batch_first=True)

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
        x = self.cnn(x)  # (batch_size*seq_len, lstm_input_size)
        # (batch_size, seq_len, lstm_input_size)
        x = x.view(batch_size, seq_len, -1)

        # Process through bidirectional LSTMs
        x, _ = self.lstm1(x)  # LSTM returns (output, (h_n, c_n))
        x = self.dropout(x)   # Apply dropout to LSTM output
        x, _ = self.lstm2(x)  # Pass through second LSTM

        # Final output layer
        return self.fc(x)  # (batch_size, seq_len, 88)
