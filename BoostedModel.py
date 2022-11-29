import torch
import torch.nn as nn
import torch.nn.functional as F

from CarVecModel import CarVecModel
from ChooseModel import ChooseModel


class BoostedModel(nn.Module):
    def __init__(self):
        super(BoostedModel, self).__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.loss = lambda y_pred, y: torch.mean(self.mse(y_pred, y), dim=1)
        self.steps = 0

        self.main = CarVecModel()
        self.boost = CarVecModel()
        self.choose = ChooseModel()

        self.fc_boost1 = nn.Linear(71, 49)
        self.fc_boost2 = nn.Linear(49, 26)
        self.fc_boost3 = nn.Linear(26, 3)

        self.fc_choose1 = nn.Linear(77, 49)
        self.fc_choose2 = nn.Linear(49, 26)
        self.fc_choose3 = nn.Linear(26, 1)

    def forward(self, x):
        y1 = self.main.forward(x)
        y2 = self.boost.forward(x)
        b = self.choose(torch.cat((x, y1, y2)))
        if b > 0.5:
            return y1
        else:
            return y2
