import torch
import torch.nn as nn
import torch.nn.functional as F


class BoolModel(nn.Module):
    def __init__(self, config, hidden_size):
        super(BoolModel, self).__init__()
        self.loss = nn.BCELoss()
        self.steps = 0

        self.fc1 = nn.Linear(config.in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
