import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from input_formater import ball_input, car_input
from output_formater import ball_output, car_output


class Auswerter:
    def __init__(self, config, device):
        self.device = device
        self.config = config
        # Contingency Tables  (x_test,  y_test,  y_pred)
        self.tables = []
        for i in range(5):
            self.tables.append(torch.zeros((2, 2, 2), device=self.device))

        # Stats
        # [b_pos, b_vel, b_ang_vel, c_pos, c_vel, c_ang_vel, c_forward, c_up] x
        # [n, min, max, sum, sum_of_squares, mean, sdv]
        self.stats = []
        for i in range(39):
            self.stats.append(torch.zeros((8, 7), device=self.device))
            self.stats[0][:, :, 1] = 1000000

    def gather_info(self, y_pred, y_test, x_test):
        n = x_test.shape[0]
        b_pos_pred, b_vel_pred, b_ang_vel_pred, c_pos_pred, c_forward_pred, c_up_pred, c_vel_pred, c_ang_vel_pred, c_on_ground_pred, c_ball_touch_pred, c_has_jump_pred, c_has_flip_pred, c_is_demo_pred = y_pred

        ball_y = ball_output(x_test, y_test)
        car_y = car_output(x_test, y_test)
        bnw = ball_near_wall(x_test)
        cnw = car_near_wall(x_test)
        bnc = ball_near_ceiling(x_test)
        cnc = car_near_ceiling(x_test)
        cncr = car_near_car(x_test)

        if isinstance(c_forward_pred, tuple):
            c_forward_pred = torch.cat(c_forward_pred, dim=1)
        if isinstance(c_up_pred, tuple):
            c_up_pred = torch.cat(c_up_pred, dim=1)

        Dists = torch.zeros((n, 8))
        Dists[:, 0] = L2(b_pos_pred, ball_y[:, 0:3])  # bpos
        Dists[:, 1] = L2(b_vel_pred, ball_y[:, 3:6])  # bvel
        Dists[:, 2] = L2(b_ang_vel_pred, ball_y[:, 6:9])  # bangvel
        Dists[:, 3] = L2(c_pos_pred, car_y[:, 0:3])  # cpos
        Dists[:, 4] = L2(c_vel_pred, car_y[:, 9:12])  # cvel
        Dists[:, 5] = L2(c_ang_vel_pred, car_y[:, 12:15])  # cangvel

        Dists[:, 6] = angle(c_forward_pred + x_test[:, 12:15], car_y[:, 3:6] + x_test[:, 12:15])  # forward
        Dists[:, 7] = angle(c_up_pred + x_test[:, 15:18], car_y[:, 6:9] + x_test[:, 15:18])  # up

        bool_preds = []
        # 0 on_ground
        bool_preds.append(torch.gt(c_on_ground_pred, 0.5).type(torch.float32))
        # 1 ball_touch
        bool_preds.append(torch.gt(c_ball_touch_pred, 0.5).type(torch.float32))
        # 2 has_jump
        bool_preds.append(torch.gt(c_has_jump_pred, 0.5).type(torch.float32))
        # 3 has_flip
        bool_preds.append(torch.gt(c_has_flip_pred, 0.5).type(torch.float32))
        # 4 is_demo
        bool_preds.append(torch.gt(c_is_demo_pred, 0.5).type(torch.float32))

        indices = []
        # 0 all
        indices.append(torch.Tensor(range(n)))
        # 1 on_ground
        indices.append(torch.nonzero(x_test[:, 25]))
        # 2 not on_ground
        indices.append(torch.nonzero(1 - x_test[:, 25]))
        # 3 ball_touch
        indices.append(torch.nonzero(x_test[:, 26]))
        # 4 not on_ground
        indices.append(torch.nonzero(1 - x_test[:, 26]))
        # 5 has_jump
        indices.append(torch.nonzero(x_test[:, 27]))
        # 6 not has_jump
        indices.append(torch.nonzero(1 - x_test[:, 27]))
        # 7 has_flip
        indices.append(torch.nonzero(x_test[:, 28]))
        # 8 not has_flip
        indices.append(torch.nonzero(1 - x_test[:, 28]))
        # 9 is_demo
        indices.append(torch.nonzero(x_test[:, 29]))
        # 10 not is_demo
        indices.append(torch.nonzero(1 - x_test[:, 29]))

        # 11 negative throttle
        indices.append(torch.nonzero(x_test[:, 85] < 0))
        # 12 zero throttle
        indices.append(torch.nonzero(x_test[:, 85] == 0))
        # 13 positive throttle
        indices.append(torch.nonzero(x_test[:, 85] > 0))

        # 14 zero steer
        indices.append(torch.nonzero(x_test[:, 86] == 0))
        # 15 not zero steer
        indices.append(torch.nonzero(1 - x_test[:, 86] != 0))

        # 16 negative pitch
        indices.append(torch.nonzero(x_test[:, 87] < 0))
        # 17 zero pitch
        indices.append(torch.nonzero(x_test[:, 87] == 0))
        # 18 positive pitch
        indices.append(torch.nonzero(x_test[:, 87] > 0))

        # 19 zero yaw
        indices.append(torch.nonzero(x_test[:, 88] == 0))
        # 20 not zero yaw
        indices.append(torch.nonzero(1 - x_test[:, 88] != 0))
        # 21 zero roll
        indices.append(torch.nonzero(x_test[:, 89] == 0))
        # 22 not zero roll
        indices.append(torch.nonzero(1 - x_test[:, 89] != 0))

        # 23 jump
        indices.append(torch.nonzero(x_test[:, 90]))
        # 24 not jump
        indices.append(torch.nonzero(1 - x_test[:, 90]))
        # 25 boost
        indices.append(torch.nonzero(x_test[:, 91]))
        # 26 not boost
        indices.append(torch.nonzero(1 - x_test[:, 91]))
        # 27 handbrake
        indices.append(torch.nonzero(x_test[:, 92]))
        # 28 not handbrake
        indices.append(torch.nonzero(1 - x_test[:, 92]))

        # 29 ball near wall
        indices.append(torch.nonzero(bnw))
        # 30 not ball near wall
        indices.append(torch.nonzero(1 - bnw))
        # 31 car near wall
        indices.append(torch.nonzero(cnw))
        # 32 not car near wall
        indices.append(torch.nonzero(1 - cnw))
        # 33 ball near ceiling
        indices.append(torch.nonzero(bnc))
        # 34 not ball near ceiling
        indices.append(torch.nonzero(1 - bnc))
        # 35 car near ceiling
        indices.append(torch.nonzero(cnc))
        # 36 not car near ceiling
        indices.append(torch.nonzero(1 - cnc))
        # 37 car near car
        indices.append(torch.nonzero(cncr))
        # 38 not car near car
        indices.append(torch.nonzero(1 - cncr))

        # Contingency Tables
        for i in range(len(bool_preds)):
            for j in range(n):
                self.tables[i][int(x_test[j, 25 + i]), int(y_test[j, 15 + i]), int(bool_preds[i][j])] += 1

        # Stats
        for i in range(len(indices)):
            self.stats[i] = generate_stats(self.stats[i], indices[i], Dists)

    # Berechnen von Arithmetischem Mittel und Standardabweichung
    def finish_stats(self):
        for i in range(len(self.stats)):
            for j in range(8):
                n = self.stats[i][j][0]
                self.stats[i][j][5] = self.stats[i][j][3] / n
                mean2 = self.stats[i][j][5] * self.stats[i][j][5]
                SSD = self.stats[i][j][4] - (n * mean2)
                Var = SSD / (n-1)
                sdv = math.sqrt(Var)
                self.stats[i][j][6] = sdv

    def save(self):
        path = self.config.result_path
        torch.save((self.tables, self.stats), path + "/Auswertung.pt")

    def load(self, path):
        self.tables, self.stats = torch.load(path)


def L2(y_pred, y_true):
    dist = nn.PairwiseDistance()
    if isinstance(y_pred, tuple):
        y_pred = torch.cat(y_pred, dim=1)
    return dist(y_pred, y_true)


def angle(y_pred, y_true):
    return torch.rad2deg(torch.acos(F.cosine_similarity(y_pred, y_true)))


def generate_stats(stat_tensor, indices, dists):
    n = indices.shape[0]
    for i in range(8):
        stat_tensor[i][0] += n
        stat_tensor[i][1] = min(stat_tensor[i][1], torch.min(dists[indices, i]).item())
        stat_tensor[i][2] = max(stat_tensor[i][2], torch.max(dists[indices, i]).item())
        stat_tensor[i][3] += torch.sum(dists[indices, i]).item()
        stat_tensor[i][4] += torch.sum(torch.square(dists[indices, i])).item()
    return stat_tensor




def ball_near_wall(x_test):
    x = x_test[:, 0]
    y = x_test[:, 1]
    # Arenawand -100
    near = x > 3996
    near = near or x < -3996
    near = near or y > 5020
    near = near or y < -5020
    return near


def ball_near_ceiling(x_test):
    z = x_test[:, 3]
    # Deckenhöhe -100
    near = z > 1944
    return near


def car_near_wall(x_test):
    x = x_test[:, 9]
    y = x_test[:, 10]
    # Arenawand -101,58
    near = x > 3994.42
    near = near or x < -3994.42
    near = near or y > 5018.42
    near = near or y < -5018.42
    return near


def car_near_ceiling(x_test):
    z = x_test[:, 11]
    # Deckenhöhe -101,58
    near = z > 1942.42
    return near


def car_near_car(x_test):
    # 2 * 101,58
    dist = L2(x_test[:, 47:50], x_test[:, 9:12])
    near = dist < 203.16
    return near
