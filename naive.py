import torch
import torch.nn as nn
import torch.nn.functional as F


class Naive(nn.Module):
    def __init__(self, config):
        super(Naive, self).__init__()

        self.fc1 = nn.Linear(config.in_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, config.out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def criterion(self, y_predicted, y_train):
        return nn.MSELoss(y_predicted, y_train)
