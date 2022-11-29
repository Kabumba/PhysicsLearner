import torch
import torch.nn as nn
import torch.nn.functional as F


class BallVecModel(nn.Module):
    def __init__(self):
        super(BallVecModel, self).__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.loss = lambda y_pred, y: torch.mean(self.mse(y_pred, y), dim=1)
        self.steps = 0

        self.fc1 = nn.Linear(43, 29)
        self.fc2 = nn.Linear(29, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x