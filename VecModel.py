import torch
import torch.nn as nn

from ScalarModel import ScalarModel


class VecModel(nn.Module):
    def __init__(self, config, input_size, hidden_size):
        super(VecModel, self).__init__()
        self.config = config
        self.loss = self.mse
        self.steps = 0

        self.models = {
            "x": ScalarModel(input_size, hidden_size),
            "y": ScalarModel(input_size, hidden_size),
            "z": ScalarModel(input_size, hidden_size),
        }

    def forward(self, x_in):
        x = self.models["x"](x_in)
        y = self.models["y"](x_in)
        z = self.models["z"](x_in)
        return x, y, z

    def mse(self, y_pred, y_train):
        x = self.models["x"].loss(y_pred[0][:, 0], y_train[:, 0])
        y = self.models["y"].loss(y_pred[1][:, 0], y_train[:, 1])
        z = self.models["z"].loss(y_pred[2][:, 0], y_train[:, 2])
        return (x + y + z)/3

    def to(self, device, non_blocking=True):
        for model in self.models:
            self.models[model].to(device, non_blocking=non_blocking)

    def init_optim(self):
        x = torch.optim.Adam(self.models["x"].parameters(), lr=self.config.learning_rate)
        y = torch.optim.Adam(self.models["y"].parameters(), lr=self.config.learning_rate)
        z = torch.optim.Adam(self.models["z"].parameters(), lr=self.config.learning_rate)
        return x, y, z
