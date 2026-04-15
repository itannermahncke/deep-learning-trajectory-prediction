"""
Class definition for an LSTM model
"""

import torch
from torch import nn


class LSTMModel(nn.Module):
    """
    A class to define an LSTM model.

    Attributes:
        repeat_times: an int representing the number of times to repeat the last hidden state.
        lstm1: the first LSTM layer.
        dropout1: the first dropout.
        lstm2: the second LSTM layer.
        dropout2: the second dropout.
        time_distributed: the time distributed dense layer.
        dense: the final dense layer.

    Methods:
        forward: a method to define the forward pass of the model.
    """

    def __init__(self, dims):
        """
        Initiate the LSTM model class

        Args:
            dims: a dictionary containing the dimensions of the model.
        """
        super().__init__()

        self.repeat_times = dims["repeat_times"]

        # Define the first LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=dims["input"],
            hidden_size=dims["hidden"],
            num_layers=dims["layer"],
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dims["dropout"])

        # Define the second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=dims["hidden"],
            hidden_size=dims["hidden"],
            num_layers=dims["layer"],
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dims["dropout"])

        # Define the TimeDistributed Dense layer
        self.time_distributed = nn.Linear(dims["hidden"], dims["hidden"])

        # Define the final Dense layer
        self.dense = nn.Linear(dims["hidden"], dims["output"])

    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
            x: a tensor containing the input data.

        Returns:
            a tensor containing the output data.
        """
        # LSTM layer 1
        x, (h, _) = self.lstm1(x)
        x = self.dropout1(x)

        # Repeat Vector
        h_last = h[-1]  # get the last hidden state
        h_last_repeated = h_last.unsqueeze(1).repeat(
            1, self.repeat_times, 1
        )  # repeat vector

        # LSTM layer 2
        x, (h, _) = self.lstm2(h_last_repeated)
        x = self.dropout2(x)

        # Apply the TimeDistributed(Dense) layer
        # Reshape x to (batch_size * seq_len, hidden_size) to apply the dense layer
        batch_size, seq_len, hidden_size = x.size()
        x = x.reshape(batch_size * seq_len, hidden_size)
        x = self.time_distributed(x)
        x = torch.tanh(x)

        # Reshape back to (batch_size, seq_len, hidden_size)
        x = x.reshape(batch_size, seq_len, hidden_size)

        # Take the output from the last time step
        x = x[:, -1, :]

        # Apply the final Dense layer
        x = self.dense(x)

        return x
