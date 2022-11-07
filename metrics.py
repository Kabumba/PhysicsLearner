import numpy
import torch
import torch.nn.functional as F


class Metrics:
    def __init__(self, configuration):
        self.config = configuration
        self.bo = 9
        if not self.config.ball_out:
            self.bo = 0

    def ball_pos_loss(self, y_pred, y_true):
        return torch.mean(torch.cdist(y_pred[:, 3], y_true[:, :3]))

    def ball_lin_vel_loss(self, y_pred, y_true):
        return torch.mean(torch.cdist(y_pred[:, 3:6], y_true[:, 3:6]))

    def ball_ang_vel_loss(self, y_pred, y_true):
        return torch.mean(torch.cdist(y_pred[:, 6:9], y_true[:, 6:9]))

    def car_pos_loss(self, y_pred, y_true):
        return torch.mean(torch.cdist(y_pred[:, self.bo + 0:self.bo + 3], y_true[:, self.bo + 0:self.bo + 3]))

    def car_forward_loss(self, y_pred, y_true):
        return torch.mean(F.cosine_similarity(y_pred[:, self.bo + 3:self.bo + 6], y_true[:, self.bo + 3:self.bo + 6]))

    def car_up_loss(self, y_pred, y_true):
        return torch.mean(F.cosine_similarity(y_pred[:, self.bo + 6:self.bo + 9], y_true[:, self.bo + 6:self.bo + 9]))

    def car_lin_vel_loss(self, y_pred, y_true):
        return torch.mean(torch.cdist(y_pred[:, self.bo + 9:self.bo + 12], y_true[:, self.bo + 9:self.bo + 12]))

    def car_ang_vel_loss(self, y_pred, y_true):
        return torch.mean(torch.cdist(y_pred[:, self.bo + 12:self.bo + 15], y_true[:, self.bo + 12:self.bo + 15]))

    def car_on_ground_acc(self, y_pred, y_true):
        og_pred = torch.gt(y_pred[:, self.bo + 16], 0.5)
        og_true = torch.gt(y_true[:, self.bo + 16], 0.5)
        og_diff = torch.abs(og_true - og_pred)
        return torch.mean(og_diff)

    def car_ball_touch_acc(self, y_pred, y_true):
        og_pred = torch.gt(y_pred[:, self.bo + 17], 0.5)
        og_true = torch.gt(y_true[:, self.bo + 17], 0.5)
        og_diff = torch.abs(og_true - og_pred)
        return torch.mean(og_diff)

    def car_has_jump_acc(self, y_pred, y_true):
        og_pred = torch.gt(y_pred[:, self.bo + 18], 0.5)
        og_true = torch.gt(y_true[:, self.bo + 18], 0.5)
        og_diff = torch.abs(og_true - og_pred)
        return torch.mean(og_diff)

    def car_has_flip_acc(self, y_pred, y_true):
        og_pred = torch.gt(y_pred[:, self.bo + 19], 0.5)
        og_true = torch.gt(y_true[:, self.bo + 19], 0.5)
        og_diff = torch.abs(og_true - og_pred)
        return torch.mean(og_diff)

    def car_is_demo_acc(self, y_pred, y_true):
        og_pred = torch.gt(y_pred[:, self.bo + 20], 0.5)
        og_true = torch.gt(y_true[:, self.bo + 20], 0.5)
        og_diff = torch.abs(og_true - og_pred)
        return torch.mean(og_diff)


def rot_matrix_from_forward_up(forward, up):
    forward_normalized = F.normalize(forward)
    up_normalized = F.normalize(up - (torch.bmm(torch.transpose(forward_normalized), up) * forward_normalized))
    rot_matrix = torch.zeros(forward.shape[0], 3, 3)
    rot_matrix[:, :, 0] = forward_normalized
    rot_matrix[:, :, 1] = torch.cross(forward_normalized, up_normalized)
    rot_matrix[:, :, 2] = up_normalized
    return rot_matrix
