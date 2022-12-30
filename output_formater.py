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


def output(x, y):
    output_dict = {}
    output_dict["bpos"] = y[:, 0:3] - x[:, 0:3]
    output_dict["bvel"] = y[:, 3:6] - x[:, 3:6]
    output_dict["bangvel"] = y[:, 6:9] - x[:, 6:9]
    output_dict["cpos"] = y[:, 9:12] - x[:, 9:12]
    output_dict["cforward"] = y[:, 12:15] - x[:, 12:15]
    output_dict["cup"] = y[:, 15:18] - x[:, 15:18]
    output_dict["cvel"] = y[:, 18:21] - x[:, 18:21]
    output_dict["cangvel"] = y[:, 21:24] - x[:, 21:24]
    output_dict["conground"] = y[:, 24]
    output_dict["cballtouch"] = y[:, 25]
    output_dict["chasjump"] = y[:, 26]
    output_dict["chasflip"] = y[:, 27]
    output_dict["cisdemo"] = y[:, 28]
    return output_dict
