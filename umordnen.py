import math
import os
import random
import shutil

from configuration import Configuration

gs_path = "C:/Users/Frederik/Masterarbeit/Data/Train"
files = os.listdir(gs_path)
for f in files:
    if f.endswith(".pt"):
        r = math.floor(random.Random().uniform(0, 3))
        shutil.move(os.path.join(gs_path, f), os.path.join(f"C:/Users/Frederik/Masterarbeit/Data/Train/Train{r}", f))
print("Done")
