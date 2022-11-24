import torch.nn as nn
import torch.nn.functional as F


class Vec3model(nn.Module):
    def __init__(self, config, hidden_size):
        super(Vec3model, self).__init__()
        self.loss = nn.MSELoss()
        self.steps = 0

        self.fc1 = nn.Linear(config.in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
