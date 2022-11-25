import torch


def ball_input(x):
    x_ball = torch.zeros(x.shape[0], 43, device=x.device)
    # absolute ball
    x_ball[:, 0:9] = x[:, 0:9]

    # relative relevant car 1
    x_ball[:, 9:12] = x[:, 9:12] - x[:, 0:3]
    x_ball[:, 12:18] = x[:, 12:18]  # rotation
    x_ball[:, 18:24] = x[:, 18:24] - x[:, 3:9]
    x_ball[:, 24] = x[:, 26]  # ball touch
    x_ball[:, 25] = x[:, 29]  # is demo

    # relative relevant car 2
    x_ball[:, 26:29] = x[:, 47:50] - x[:, 0:3]
    x_ball[:, 29:35] = x[:, 50:56]  # rotation
    x_ball[:, 35:41] = x[:, 56:62] - x[:, 3:9]
    x_ball[:, 41] = x[:, 64]  # ball touch
    x_ball[:, 42] = x[:, 67]  # is demo
    return x_ball


def car_input(x):
    x_car = torch.zeros(x.shape[0], 71, device=x.device)
    # relative ball
    x_car[:, 0:3] = x[:, 0:3] - x[:, 9:12]
    x_car[:, 3:6] = x[:, 3:6] - x[:, 18:24]

    # absolute main car
    x_car[:, 9:47] = x[:, 9:47]
    x_car[:, 47:55] = x[:, 85:93]  # inputs

    # relative relevant enemy car
    x_car[:, 55:70] = x[:, 47:62] - x[:, 9:24]
    x_car[:, 70] = x[:, 67]  # is demo
    return x_car
