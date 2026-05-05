# example from here: https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-networks-using-pytorch/

import torch
from torch import nn


class SimpleLSTMModel(nn.Module):
    def __init__(self, hyperparameters):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_dim = hyperparameters["hidden"]
        self.layer_dim = hyperparameters["layer"]
        self.lstm = nn.LSTM(
            hyperparameters["input"], self.hidden_dim, self.layer_dim, batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, hyperparameters["output"])

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
