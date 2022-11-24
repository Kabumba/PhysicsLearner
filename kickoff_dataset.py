import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

import IO_manager
import mirror_data
from configuration import Configuration
from logger import log


class ObservationTransformer:
    def __init__(self, config: Configuration):
        self.config = config
        self.n_factor = self.n_factor()

    def n_factor(self):
        factor = 1
        if self.config.mirror:
            factor += 1
        if self.config.invert:
            factor += 1
        if self.config.mirror_and_invert:
            factor += 1
        if self.config.num_car_in > 0 or self.config.num_car_out > 0:
            factor *= 2
        return factor

    def __call__(self, sample, mode):
        x, y = sample
        x_new = torch.zeros(self.config.in_size, device=x.device)
        y_new = torch.zeros(self.config.out_size, device=y.device)
        fci = 0  # first car index input
        fco = 0  # first car index output
        dbi = self.config.num_car_in * 46 # index where delta ball obs start

        # car swap
        swap_cars = False
        if self.config.num_car_in > 0 or self.config.num_car_out > 0:
            swap_cars = mode >= self.n_factor / 2
            if swap_cars:
                mode -= self.n_factor / 2

        # mirror operations
        if mode > 0:
            if self.config.mirror and mode == 1:
                x = mirror_data.mirror_state(x)
                y = mirror_data.mirror_state(y)
            elif self.config.mirror_and_invert and mode == (self.n_factor / 2 - 1):
                x = mirror_data.invert_and_mirror_state(x)
                y = mirror_data.invert_and_mirror_state(y)
            elif self.config.invert:
                x = mirror_data.invert_state(x)
                y = mirror_data.invert_state(y)

        # inputs

        if self.config.ball_in:
            x_new[:9] = x[:9]
            fci = 9
            dbi += 9
        else:
            if self.config.delta_inputs:
                fci = 9
        if self.config.num_car_in == 1:
            if swap_cars:
                x_new[fci:fci + 38] = x[47:85]
                x_new[fci + 38: fci + 46] = x[93:]
            else:
                x_new[fci:fci + 38] = x[9:47]
                x_new[fci + 38: fci + 46] = x[85:93]
        if self.config.num_car_in == 2:
            if swap_cars:
                x_new[fci:fci + 38] = x[47:85]
                x_new[fci + 38:fci + 76] = x[9:47]
                x_new[fci + 76:fci + 84] = x[93:]
                x_new[fci + 84:fci + 92] = x[85:93]
            else:
                x_new[fci:fci + 92] = x[9:]
            x_new[fci + 38:fci + 53] -= x_new[9:24]
        if self.config.ball_in:
            if self.config.delta_inputs:
                x_new[dbi:dbi + 3] = x[:3] - x_new[9:12]
                x_new[dbi + 3:dbi + 9] = x[3:9] - x_new[18:24]
        else:
            if self.config.delta_inputs:
                fci = 9
                x_new[:3] = x[:3]-x_new[9:12]
                x_new[3:9] = x[3:9] - x_new[18:24]

        # targets
        if self.config.ball_out:
            y_new[:9] = y[:9]
            fco = 9
            if self.config.delta_targets:
                y_new[:9] -= x_new[:9]
        if self.config.num_car_out >= 1:
            if swap_cars:
                y_new[fco:fco + 15] = y[47:62]
                # skip over boost-amount
                y_new[fco + 15:fco + 20] = y[63:68]
                # leave out time related data
            else:
                y_new[fco:fco + 15] = y[9:24]
                # skip over boost-amount
                y_new[fco + 15:fco + 20] = y[25:30]
                # leave out time related data
            if self.config.delta_targets:
                y_new[fco:fco + 3] -= x_new[fci:fci+3]
                y_new[fco+3:fco + 9] -= x_new[fci+3:fci + 9]
                y_new[fco + 9:fco + 15] -= x_new[fci + 9:fci + 15]
        if self.config.num_car_out == 2:
            if swap_cars:
                y_new[fco + 20:fco + 35] = y[9:24]
                # skip over boost-amount
                y_new[fco + 35:fco + 40] = y[25:30]
                # leave out time related data
            else:
                y_new[fco + 20:fco + 35] = y[47:62]
                # skip over boost-amount
                y_new[fco + 35:fco + 40] = y[63:68]
                # leave out time related data
            if self.config.delta_targets:
                y_new[fco + 20:fco + 23] -= x_new[fci + 47:fci + 50]
                y_new[fco + 23:fco + 29] -= x_new[fci + 50:fci + 56]
                y_new[fco + 29:fco + 35] -= x_new[fci + 56:fci + 62]
        return x_new, y_new


class KickoffDataset(Dataset):

    def __init__(self, data_file, config: Configuration):
        # data loading
        self.game_states = torch.load(data_file)  # , map_location=lambda storage, loc: storage.cuda(0))
        normalize(config, self.game_states)
        # log(f'self.game_states {self.game_states.device}')
        self.transform = ObservationTransformer(config)
        self.n_samples = self.game_states.shape[0] - 1
        self.n_factor = 1
        if self.transform:
            self.n_factor = self.transform.n_factor
            self.n_samples = self.n_factor * self.n_samples

    def __getitem__(self, index):
        i = math.floor(index / self.n_factor)
        x = self.game_states[i]
        y = self.game_states[i + 1][:85]
        sample = x, y
        if self.transform:
            mode = index % self.n_factor
            sample = self.transform(sample, mode)
        # log(f'sample {(sample[0].device, sample[1].device)}')
        return sample

    def __len__(self):
        return self.n_samples


class KickoffEnsemble(Dataset):
    def __init__(self, data_dir, config: Configuration):
        self.config = config
        log("Loading Data into RAM...")
        self.kickoffs = [KickoffDataset((data_dir + "/" + file), self.config) for file in os.listdir(data_dir)]
        log(f"Data Device: {self.kickoffs[0].game_states.device}")
        self.n_samples = np.sum(np.array([k.n_samples for k in self.kickoffs]))
        self.indices = np.zeros((len(self.kickoffs),))
        self.indices[0] = self.kickoffs[0].n_samples
        for i in range(len(self.kickoffs) - 1):
            self.indices[i + 1] = self.indices[i] + self.kickoffs[i + 1].n_samples
        log(f'Loaded {len(self.kickoffs)} files containing {self.n_samples} observations')

    def __getitem__(self, index):
        kickoff_index = self.binary_search(0, len(self.kickoffs) - 1, index)
        new_index = index
        if kickoff_index > 0:
            new_index -= self.indices[kickoff_index - 1]
        return self.kickoffs[kickoff_index][new_index]

    def binary_search(self, start: int, end: int, value):
        if start == end:
            return start
        mid = int(start + (end - start) / 2)
        if value >= self.indices[mid]:
            return self.binary_search(mid + 1, end, value)
        if value < self.indices[mid]:
            return self.binary_search(start, mid, value)

    def __len__(self):
        return self.n_samples


def normalize(config, game_states):
    coefs = torch.tensor([
        # ball
        1 / config.ball_pos_norm_factor,
        1 / config.ball_pos_norm_factor,
        1 / config.ball_pos_norm_factor,  # pos
        1 / config.ball_vel_norm_factor,
        1 / config.ball_vel_norm_factor,
        1 / config.ball_vel_norm_factor,  # vel
        1 / config.ball_ang_vel_norm_factor,
        1 / config.ball_ang_vel_norm_factor,
        1 / config.ball_ang_vel_norm_factor,  # ang_vel
        # car1
        1 / config.car_pos_norm_factor,
        1 / config.car_pos_norm_factor,
        1 / config.car_pos_norm_factor,  # pos
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,  # forward
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,  # up
        1 / config.car_vel_norm_factor,
        1 / config.car_vel_norm_factor,
        1 / config.car_vel_norm_factor,  # vel
        1 / config.car_ang_vel_norm_factor,
        1 / config.car_ang_vel_norm_factor,
        1 / config.car_ang_vel_norm_factor,  # ang_vel
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # bools and timings
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,  # forward flip
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,  # up flip
        1 / config.car_vel_norm_factor,
        1 / config.car_vel_norm_factor,
        1 / config.car_vel_norm_factor,  # vel flip
        1,  # pitch flip
        1,  # yaw + roll flip
        # car2
        1 / config.car_pos_norm_factor,
        1 / config.car_pos_norm_factor,
        1 / config.car_pos_norm_factor,  # pos
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,  # forward
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,  # up
        1 / config.car_vel_norm_factor,
        1 / config.car_vel_norm_factor,
        1 / config.car_vel_norm_factor,  # vel
        1 / config.car_ang_vel_norm_factor,
        1 / config.car_ang_vel_norm_factor,
        1 / config.car_ang_vel_norm_factor,  # ang_vel
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # bools and timings
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,  # forward flip
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,
        1 / config.car_ang_norm_factor,  # up flip
        1 / config.car_vel_norm_factor,
        1 / config.car_vel_norm_factor,
        1 / config.car_vel_norm_factor,  # vel flip
        1,  # pitch flip
        1,  # yaw + roll flip
        # inputs
        1, 1, 1, 1, 1, 1, 1, 1,  # car1
        1, 1, 1, 1, 1, 1, 1, 1  # car2
    ], device=game_states.device)
    game_states[:, :85] = game_states[:, :85] * coefs[:85]
