import torch
import torch.nn as nn
import torch.nn.functional as F


class CarBoolModel(nn.Module):
    def __init__(self, config):
        super(CarBoolModel, self).__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.loss = lambda y_pred, y: torch.mean(self.bce(y_pred, y), dim=1)
        self.config = config
        self.steps = 0

        self.fc1 = nn.Linear(71, 47)
        self.fc2 = nn.Linear(47, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def init_optim(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)