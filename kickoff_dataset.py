import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from configuration import Configuration
from logger import log
from observation_transformer import ObservationTransformer


class KickoffDataset(Dataset):

    def __init__(self, data_file, config: Configuration, device):
        # data loading
        self.game_states = torch.load(data_file)
        self.game_states.to(device)
        self.config = config
        normalize(config, self.game_states)
        # log(f'self.game_states {self.game_states.device}')
        self.transform = ObservationTransformer(config)
        self.n_samples = self.game_states.shape[0] - 1
        self.n_factor = 1
        if self.transform:
            self.n_factor = self.transform.n_factor
            self.n_samples = self.n_factor * self.n_samples
        # self.losses = torch.ones(self.n_samples)
        # self.losses.mul_(3000000000.0)
        # base_name = os.path.splitext(os.path.basename(data_file))[0]
        # torch.save(os.path.join(config.loss_path, base_name + ".loss"))

    def set_new_config(self, config):
        unnormalize(self.config, self.game_states)
        self.config = config
        normalize(config, self.game_states)
        self.transform = ObservationTransformer(config)
        self.n_factor = 1
        if self.transform:
            self.n_factor = self.transform.n_factor
            self.n_samples = self.n_factor * self.n_samples

    def __getitem__(self, index):

        i = math.floor(index / self.n_factor)
        try:
            x = self.game_states[i]
            y = self.game_states[i + 1][:85]
            sample = x, y
            if self.transform:
                mode = index % self.n_factor
                sample = self.transform(sample, mode)
            # log(f'sample {(sample[0].device, sample[1].device)}')
        except IndexError:
            log(f"Shape: {self.game_states.shape}")
            log(f"index: {index}")
            log(f"i: {i}")
            raise
        return sample

    def __len__(self):
        return self.n_samples


class KickoffEnsemble(Dataset):
    def __init__(self, data_dir, partition, config: Configuration, device):
        self.config = config
        log(f"Loading new partition Data into RAM...")
        if partition is None:
            partition = os.listdir(data_dir)
        self.kickoffs = [KickoffDataset((data_dir + "/" + file), self.config, device) for file in partition]
        # log(f"Data Device: {self.kickoffs[0].game_states.device}")
        self.n_samples = torch.sum(torch.tensor([k.n_samples for k in self.kickoffs])).item()
        self.indices = torch.zeros(len(self.kickoffs))
        self.indices[0] = self.kickoffs[0].n_samples
        for i in range(len(self.kickoffs) - 1):
            self.indices[i + 1] = self.indices[i] + self.kickoffs[i + 1].n_samples
        log(f'Loaded {len(self.kickoffs)} files containing {self.n_samples} observations')

    def __getitem__(self, index):

        index = int(index)
        kickoff_index = self.binary_search(0, len(self.kickoffs) - 1, index)
        new_index = index
        if kickoff_index > 0:
            new_index -= self.indices[kickoff_index - 1]
        try:
            return self.kickoffs[kickoff_index][new_index], index
        except IndexError:
            log(f"outer_index: {index}")
            log(f"kickoff_index: {kickoff_index}")
            log(f"new_index: {new_index}")
            raise

    def binary_search(self, start: int, end: int, value):
        if start == end:
            return start
        mid = int(start + (end - start) / 2)
        if value >= self.indices[mid]:
            return self.binary_search(mid + 1, end, value)
        if value < self.indices[mid]:
            return self.binary_search(start, mid, value)

    def set_new_config(self, config):
        self.config = config
        for kickoff in self.kickoffs:
            kickoff.set_new_config(config)

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

def unnormalize(config, game_states):
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
    game_states[:, :85] = game_states[:, :85] / coefs[:85]
