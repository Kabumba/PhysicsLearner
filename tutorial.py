import math
import os

import torch
from torch.utils.data import DataLoader

import configuration
from kickoff_dataset import KickoffDataset

x = 1
config = configuration.Configuration("Beispiel")
if not os.path.exists(config.checkpoint_path):
    os.mkdir(config.checkpoint_path)
torch.save(x, config.checkpoint_path + "/lol.pt")
