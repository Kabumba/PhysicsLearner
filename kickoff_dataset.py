import numpy as np
import torch
from torch.utils.data import Dataset

import IO_manager


class KickoffDataset(Dataset):

    def __init__(self, data_file, transform=None):
        # data loading
        self.game_states = torch.load(data_file)
        self.n_samples = self.game_states.shape[0] - 1
        self.transform = transform

    def __getitem__(self, index):
        x = self.game_states[index]
        y = self.game_states[index + 1][:85]
        sample = x, y
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples
