import torch
import torch.nn as nn
import torch.nn.functional as F


class ChooseModel(nn.Module):
    def __init__(self):
        super(ChooseModel, self).__init__()
        self.bce = nn.BCELoss(reduction="none")
        self.loss = lambda y_pred, y: torch.mean(self.bce(y_pred, y), dim=1)
        self.steps = 0

        self.fc1 = nn.Linear(77, 47)
        self.fc2 = nn.Linear(47, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x