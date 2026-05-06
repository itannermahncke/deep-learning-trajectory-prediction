# example from here: https://www.scaler.com/topics/pytorch/lstm-pytorch/

import torch
from torch import nn


class SimpleBiLSTM(nn.Module):
    def __init__(self, hyperparameters):
        super(SimpleBiLSTM, self).__init__()
        self.hidden_dim = hyperparameters["hidden"]
        self.layer_dim = hyperparameters["layer"]
        self.lstm = nn.LSTM(
            hyperparameters["input"],
            self.hidden_dim,
            self.layer_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(self.hidden_dim * 2, hyperparameters["output"])

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
