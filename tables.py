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

print(auswerter.tables[0][0][0][0])
accs = []
for i in range(5):
    print(f"-----{i}-------")
    accs.append(torch.zeros((2, 2, 2), device=auswerter.device))
    for x in range(2):
        for y in range(2):
            accs[i][x, y, 1] = torch.sum(auswerter.tables[i][x, y, :]).item()
            accs[i][x, y, 0] = 100*auswerter.tables[i][x, y, y]/accs[i][x, y, 1]
            print(f"\\hline")
            print(f"${x}$ & ${y}$ & ${accs[i][x,y,0]:.4f}$\\% & ${int(accs[i][x,y,1])}$ \\\\")
    n = torch.sum(auswerter.tables[i][:, :, :]).item()
    print(f"n: {n}")
print("done")