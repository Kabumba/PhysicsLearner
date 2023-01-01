import os

import torch

from IO_manager import Directories
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from auswertung import Auswerter

res_dir = "../Ergebnisse/Baseline/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
auswerter = Auswerter(None, device)
auswerter.load(res_dir + "Auswertung.pt")

ns = []
for i in range(39):
    print(f"-----{i}-------")
    ns.append(int(auswerter.stats[i][0, 0]))
    print(f"${100*ns[i]/ns[0]:.2f}$\\%")
print("done")
