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
names = [
    "Wahr",
    "g_{t}^{(1)}",
    "\\neg g_{t}^{(1)}",
    "T_{t}^{(1)}",
    "\\neg T_{t}^{(1)}",
    "j_{t}^{(1)}",
    "\\neg j_{t}^{(1)}",
    "f_{t}^{(1)}",
    "\\neg f_{t}^{(1)}",
    "d_{t}^{(1)}",
    "\\neg d_{t}^{(1)}",
    "\\overline{t}_{t}^{(1)} < 0",
    "\\overline{t}_{t}^{(1)} = 0",
    "\\overline{t}_{t}^{(1)} > 0",
    "\\overline{s}_{t}^{(1)} = 0",
    "\\overline{s}_{t}^{(1)} \\neq 0",
    "\\overline{p}_{t}^{(1)} < 0",
    "\\overline{p}_{t}^{(1)} = 0",
    "\\overline{p}_{t}^{(1)} > 0",
    "\\overline{y}_{t}^{(1)} = 0",
    "\\overline{y}_{t}^{(1)} \\neq 0",
    "\\overline{r}_{t}^{(1)} = 0",
    "\\overline{r}_{t}^{(1)} \\neq 0",
    "\\overline{j}_{t}^{(1)}",
    "\\neg \\overline{j}_{t}^{(1)}",
    "\\overline{b}_{t}^{(1)}",
    "\\neg \\overline{b}_{t}^{(1)}",
    "\\overline{h}_{t}^{(1)}",
    "\\neg \\overline{h}_{t}^{(1)}",
    "\\text{bnw}_{t}",
    "\\neg \\text{bnw}_{t}",
    "\\text{cnw}_{t}",
    "\\neg \\text{cnw}_{t}",
    "\\text{bnc}_{t}",
    "\\neg \\text{bnc}_{t}",
    "\\text{cnc}_{t}",
    "\\neg \\text{cnc}_{t}",
    "\\text{cc}_{t}",
    "\\neg \\text{cc}_{t}",
]
lines = [0, 2, 4, 6, 8, 10, 13, 15, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
lines2 = [0, 10, 28]
i = 0
print(f"Kriterium & Mittel & Stadandardabweichung & Minimum & Maximum \\\\")
print("\\hline")
for j in range(39):
    s = auswerter.stats[j][i]
    print(f"${names[j]}$ & ${s[5]:.5f}$ & ${s[6]:.4f}$ & ${s[1]:.4f}$ & ${s[2]:.4f}$\\\\")
    if j in lines:
        print("\\hline")
    if j in lines2:
        print("\\hline")

