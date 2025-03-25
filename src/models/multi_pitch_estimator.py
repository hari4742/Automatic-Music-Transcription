import torch
import torch.nn as nn


class MultiPitchEstimator(nn.Module):
    def __init__(self,
                 kernel1_size: tuple = (10, 2),
                 out_channels1: int = 32,
                 max_pool_kernel1: tuple = (4, 2),
                 kernel2_size: tuple = (3, 2),
                 out_channels2: int = 64,
                 max_pool_kernel2: tuple = (2, 1),
                 lstm1_hidden_size: int = 500,
                 dropout_size: int = 0.75,
                 lstm2_hidden_size: int = 200
                 ):
        """
        Initialize the multi-pitch estimation model with additional CNN layer.
        """
        super().__init__()

        input_shape = (288, 5)
        # Parameters for the first CNN layer
        # kernel1_size = (10, 2)
        # out_channels1 = 32
        # max_pool_kernel1 = (4, 2)
        # Parameters for the second CNN layer
        # kernel2_size = (3, 2)
        # out_channels2 = 64
        # max_pool_kernel2 = (2, 1)

        # CNN for spectrogram feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, out_channels=out_channels1, kernel_size=kernel1_size),
            nn.ReLU(),
            nn.MaxPool2d(max_pool_kernel1),
            nn.Conv2d(out_channels1, out_channels=out_channels2,
                      kernel_size=kernel2_size),
            nn.ReLU(),
            nn.MaxPool2d(max_pool_kernel2),
            nn.Flatten(),
        )

        # Calculate LSTM input size after two CNN layers
        h, w = input_shape
        # First Conv and Pool
        h = (h - kernel1_size[0] + 1)  # Conv output height
        w = (w - kernel1_size[1] + 1)  # Conv output width
        h = h // max_pool_kernel1[0]    # MaxPool output height
        w = w // max_pool_kernel1[1]    # MaxPool output width
        # Second Conv and Pool
        h = (h - kernel2_size[0] + 1)   # Conv output height
        w = (w - kernel2_size[1] + 1)   # Conv output width
        h = h // max_pool_kernel2[0]    # MaxPool output height
        w = w // max_pool_kernel2[1]    # MaxPool output width

        lstm1_input_size = out_channels2 * h * w
        # lstm1_hidden_size = 500
        # dropout_size = 0.75
        lstm2_input_size = lstm1_hidden_size * 2  # hidden size * 2 bidirectional
        # lstm2_hidden_size = 200

        fc_input_size = lstm2_hidden_size * 2

        # Bidirectional LSTMs
        self.lstm1 = nn.LSTM(input_size=lstm1_input_size, hidden_size=lstm1_hidden_size,
                             bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout_size)

        self.lstm2 = nn.LSTM(input_size=lstm2_input_size, hidden_size=lstm2_hidden_size,
                             bidirectional=True, batch_first=True)

        # Final output layer
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 88),
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
