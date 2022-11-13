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
    def __init__(self, ball_in: bool = True, ball_out: bool = True, num_car_in: int = 2, num_car_out: int = 2,
                 mirror: bool = True, invert: bool = True, mirror_and_invert: bool = True, ):
        self.num_car_in = max(0, min(2, num_car_in))
        self.num_car_out = max(0, min(2, num_car_out))
        self.mirror_and_invert = mirror_and_invert
        self.invert = invert
        self.mirror = mirror
        self.ball_out = ball_out
        self.ball_in = ball_in
        self.n_factor = self.n_factor()
        self.in_size = 0
        self.out_size = 0
        if self.ball_in:
            self.in_size += 9
        self.in_size += self.num_car_in * 46
        if self.ball_out:
            self.out_size += 9
        self.out_size += self.num_car_out * 20

    def n_factor(self):
        factor = 1
        if self.mirror:
            factor += 1
        if self.invert:
            factor += 1
        if self.mirror_and_invert:
            factor += 1
        if self.num_car_in > 0 or self.num_car_out > 0:
            factor *= 2
        return factor

    def __call__(self, sample, mode):
        x, y = sample

        x_new = torch.zeros(self.in_size)
        y_new = torch.zeros(self.out_size)
        fci = 0  # first car index
        swap_cars = False
        if self.num_car_in > 0 or self.num_car_out > 0:
            swap_cars = mode >= self.n_factor / 2
            if swap_cars:
                mode -= self.n_factor / 2
        if mode > 0:
            if self.mirror and mode == 1:
                x = mirror_data.mirror_state(x)
                y = mirror_data.mirror_state(y)
            elif self.mirror_and_invert and mode == (self.n_factor / 2 - 1):
                x = mirror_data.invert_and_mirror_state(x)
                y = mirror_data.invert_and_mirror_state(y)
            elif self.invert:
                x = mirror_data.invert_state(x)
                y = mirror_data.invert_state(y)
        if self.ball_in:
            x_new[:9] = x[:9]
            fci = 9
        if self.num_car_in == 1:
            if swap_cars:
                x_new[fci:fci + 38] = x[47:85]
                x_new[fci + 38:] = x[93:]
            else:
                x_new[fci:fci + 38] = x[9:47]
                x_new[fci + 38:] = x[85:93]
        if self.num_car_in == 2:
            if swap_cars:
                x_new[fci:fci + 38] = x[47:85]
                x_new[fci + 38:fci + 76] = x[9:47]
                x_new[fci + 76:fci + 84] = x[93:]
                x_new[fci + 84:] = x[85:93]
            else:
                x_new[fci:] = x[9:]
        if self.ball_out:
            y_new[:9] = y[:9]
        if self.num_car_out >= 1:
            if swap_cars:
                y_new[fci:fci + 15] = y[47:62]
                # skip over boost-amount
                y_new[fci + 15:fci + 20] = y[63:68]
                # leave out time related data
            else:
                y_new[fci:fci + 15] = y[9:24]
                # skip over boost-amount
                y_new[fci + 15:fci + 20] = y[25:30]
                # leave out time related data
        if self.num_car_out == 2:
            if swap_cars:
                y_new[fci + 20:fci + 35] = y[9:24]
                # skip over boost-amount
                y_new[fci + 35:fci + 40] = y[25:30]
                # leave out time related data
            else:
                y_new[fci + 20:fci + 35] = y[47:62]
                # skip over boost-amount
                y_new[fci + 35:fci + 40] = y[63:68]
                # leave out time related data

        return x_new, y_new


class KickoffDataset(Dataset):

    def __init__(self, data_file, transform: ObservationTransformer = None):
        # data loading
        self.game_states = torch.load(data_file, map_location=lambda storage, loc: storage.cuda(0))
        # log(f'self.game_states {self.game_states.device}')
        self.transform = transform
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
        self.transform = ObservationTransformer(ball_in=config.ball_in, ball_out=config.ball_out,
                                                num_car_out=config.num_car_out, num_car_in=config.num_car_in,
                                                mirror=config.mirror, invert=config.invert,
                                                mirror_and_invert=config.mirror_and_invert)
        log("Loading Data into RAM...")
        self.kickoffs = [KickoffDataset((data_dir + "/" + file), self.transform) for file in os.listdir(data_dir)]
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
