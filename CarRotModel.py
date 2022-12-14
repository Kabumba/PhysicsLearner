import torch
import torch.nn as nn
import torch.nn.functional as F


class CarRotModel(nn.Module):
    def __init__(self, config):
        super(CarRotModel, self).__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.loss = lambda y_pred, y: torch.mean(self.mse(y_pred, y), dim=1)
        self.config = config
        self.steps = 0

        self.fc1 = nn.Linear(71, 52)
        self.fc2 = nn.Linear(52, 29)
        self.fc3 = nn.Linear(29, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_optim(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)