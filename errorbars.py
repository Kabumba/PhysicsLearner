import os

import torch

from IO_manager import Directories
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from auswertung import Auswerter
from configuration import Configuration

res_dir = "../Ergebnisse/Baseline/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = Configuration("BaselineTest")
auswerter = Auswerter(config, device)
auswerter.load(res_dir + "Auswertung.pt")
# auswerter.finish_stats()
# auswerter.save()
ns = []
i=0
print(f"-----{i}-------")
x = [auswerter.stats[a][i, 5] for a in range(39)]
y = range(39)
errors = [[min(auswerter.stats[a][i, 6], auswerter.stats[a][i, 5] - auswerter.stats[a][i, 1]) for a in range(39)],
          [auswerter.stats[a][i, 6] for a in range(39)]]

plt.figure()
plt.errorbar(x, y, xerr=errors, fmt='o', color='k')
plt.yticks(range(0,40), ('', 'x3', 'x2', 'x1', ''))
plt.show()
print("done")
