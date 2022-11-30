import torch
import torch.nn as nn
import torch.nn.functional as F


class BoolModel(nn.Module):
    def __init__(self, config, input_size, hidden_size):
        super(BoolModel, self).__init__()
        self.loss = nn.BCELoss(reduction="none")
        self.steps = 0
        self.config = config

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def init_optim(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)