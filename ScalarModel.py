import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ScalarModel, self).__init__()
        self.loss = nn.MSELoss(reduction="none")

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
