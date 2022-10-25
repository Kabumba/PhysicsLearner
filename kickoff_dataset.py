import numpy as np
import torch
from torch.utils.data import Dataset

import IO_manager


class KickoffDataset(Dataset):

    def __init__(self, sequence_file, transform = None):
        # data loading
        state_list = IO_manager.load_game_state_sequence(sequence_file=sequence_file)
        game_state_length = state_list[0].shape[0]-16
        game_states = np.array([state_list[i][:game_state_length] for i in range(len(state_list))])
        self.game_states = torch.tensor(game_states, dtype=torch.float32)
        inputs = np.array([state_list[i][game_state_length:] for i in range(len(state_list) - 1)])
        self.player_inputs = torch.tensor(inputs, dtype=torch.float32)
        self.n_samples = self.player_inputs.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        pre_state = self.game_states[index]
        player_input = self.player_inputs[index]
        x = torch.concat((pre_state, player_input), dim=0)
        y = self.game_states[index + 1]
        sample = x, y
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples
