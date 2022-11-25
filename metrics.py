import sys

import numpy
import torch
import torch.nn.functional as F
import torch.nn as nn


class Metrics:
    def __init__(self, configuration):
        self.config = configuration
        self.bo = 9
        if not self.config.ball_out:
            self.bo = 0

    def ball_pos_loss(self, y_pred, y_true):
        dist = nn.PairwiseDistance()
        return torch.mean(dist(y_pred[:, 0:3] * self.config.ball_pos_norm_factor,
                               y_true[:, 0:3] * self.config.ball_pos_norm_factor)).item()

    def ball_lin_vel_loss(self, y_pred, y_true):
        dist = nn.PairwiseDistance()
        return torch.mean(dist(y_pred[:, 3:6] * self.config.ball_vel_norm_factor,
                               y_true[:, 3:6] * self.config.ball_vel_norm_factor)).item()

    def ball_ang_vel_loss(self, y_pred, y_true):
        dist = nn.PairwiseDistance()
        return torch.mean(dist(y_pred[:, 6:9] * self.config.ball_ang_vel_norm_factor,
                               y_true[:, 6:9] * self.config.ball_ang_vel_norm_factor)).item()

    def car_pos_loss(self, y_pred, y_true):
        dist = nn.PairwiseDistance()
        return torch.mean(dist(y_pred[:, self.bo + 0:self.bo + 3] * self.config.car_pos_norm_factor,
                               y_true[:, self.bo + 0:self.bo + 3] * self.config.car_pos_norm_factor)).item()

    def car_forward_loss(self, y_pred, y_true):
        return torch.mean(F.cosine_similarity(y_pred[:, self.bo + 3:self.bo + 6] * self.config.car_ang_norm_factor,
                                              y_true[:,
                                              self.bo + 3:self.bo + 6] * self.config.car_ang_norm_factor)).item()

    def car_up_loss(self, y_pred, y_true):
        return torch.mean(F.cosine_similarity(y_pred[:, self.bo + 6:self.bo + 9] * self.config.car_ang_norm_factor,
                                              y_true[:,
                                              self.bo + 6:self.bo + 9] * self.config.car_ang_norm_factor)).item()

    def car_lin_vel_loss(self, y_pred, y_true):
        dist = nn.PairwiseDistance()
        return torch.mean(dist(y_pred[:, self.bo + 9:self.bo + 12] * self.config.car_vel_norm_factor,
                               y_true[:, self.bo + 9:self.bo + 12] * self.config.car_vel_norm_factor)).item()

    def car_ang_vel_loss(self, y_pred, y_true):
        dist = nn.PairwiseDistance()
        return torch.mean(dist(y_pred[:, self.bo + 12:self.bo + 15] * self.config.car_ang_vel_norm_factor,
                               y_true[:, self.bo + 12:self.bo + 15] * self.config.car_ang_vel_norm_factor)).item()

    def car_on_ground_acc(self, y_pred, y_true):
        og_pred = torch.gt(y_pred[:, self.bo + 15], 0.5).type(torch.float32)
        og_true = torch.gt(y_true[:, self.bo + 15], 0.5).type(torch.float32)
        og_diff = torch.abs(og_true - og_pred)
        return torch.mean(1 - og_diff).item()

    def car_ball_touch_acc(self, y_pred, y_true):
        og_pred = torch.gt(y_pred[:, self.bo + 16], 0.5).type(torch.float32)
        og_true = torch.gt(y_true[:, self.bo + 16], 0.5).type(torch.float32)
        og_diff = torch.abs(og_true - og_pred)
        return torch.mean(1 - og_diff).item()

    def car_has_jump_acc(self, y_pred, y_true):
        og_pred = torch.gt(y_pred[:, self.bo + 17], 0.5).type(torch.float32)
        og_true = torch.gt(y_true[:, self.bo + 17], 0.5).type(torch.float32)
        og_diff = torch.abs(og_true - og_pred)
        return torch.mean(1 - og_diff).item()

    def car_has_flip_acc(self, y_pred, y_true):
        og_pred = torch.gt(y_pred[:, self.bo + 18], 0.5).type(torch.float32)
        og_true = torch.gt(y_true[:, self.bo + 18], 0.5).type(torch.float32)
        og_diff = torch.abs(og_true - og_pred)
        return torch.mean(1 - og_diff).item()

    def car_is_demo_acc(self, y_pred, y_true):
        og_pred = torch.gt(y_pred[:, self.bo + 19], 0.5).type(torch.float32)
        og_true = torch.gt(y_true[:, self.bo + 19], 0.5).type(torch.float32)
        og_diff = torch.abs(og_true - og_pred)
        return torch.mean(1 - og_diff).item()

    def euclid(self, y_pred, y_true):
        dist = nn.PairwiseDistance()
        return torch.mean(
            dist(y_pred * self.config.ball_pos_norm_factor, y_true * self.config.ball_pos_norm_factor)).item()

    def cos_sim(self, y_pred, y_true):
        return torch.mean(F.cosine_similarity(y_pred * self.config.car_ang_norm_factor,
                                              y_true * self.config.car_ang_norm_factor)).item()

    def acc(self, y_pred, y_true):
        og_pred = torch.gt(y_pred, 0.5).type(torch.float32)
        og_true = torch.gt(y_true, 0.5).type(torch.float32)
        og_diff = torch.abs(og_true - og_pred)
        mean = torch.mean(1 - og_diff)
        return mean.item()


def rot_matrix_from_forward_up(forward, up):
    forward_normalized = F.normalize(forward)
    up_normalized = F.normalize(up - (torch.bmm(torch.transpose(forward_normalized), up) * forward_normalized))
    rot_matrix = torch.zeros(forward.shape[0], 3, 3)
    rot_matrix[:, :, 0] = forward_normalized
    rot_matrix[:, :, 1] = torch.cross(forward_normalized, up_normalized)
    rot_matrix[:, :, 2] = up_normalized
    return rot_matrix
