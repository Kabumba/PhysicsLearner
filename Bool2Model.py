import torch
import torch.nn as nn
import torch.nn.functional as F


class Bool2Model(nn.Module):
    def __init__(self, config, hidden_size):
        super(Bool2Model, self).__init__()
        self.loss = nn.BCELoss()
        self.steps = 0

        self.fc1 = nn.Linear(config.in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x