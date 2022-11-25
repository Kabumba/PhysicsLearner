import torch


def ball_output(x, y):
    y_ball = torch.zeros(y.shape[0], 9, device=y.device)
    # delta ball
    y_ball[:, 0:9] = y[:, 0:9] - x[:, 0:9]
    return y_ball


def car_output(x, y):
    y_car = torch.zeros(y.shape[0], 20, device=y.device)
    # delta car
    y_car[:, 0:15] = y[:, 9:24] - x[:, 9:24]
    y_car[:, 15:20] = y[:, 24:29]
    return y_car
