import os
import random
import shutil

from configuration import Configuration

gs_path = "C:/Users/Frederik/Masterarbeit/Data/GameStates"
files = os.listdir(gs_path)
for f in files:
    if random.Random().uniform(0, 1) < 0.2:
        shutil.copy(os.path.join(gs_path, f), os.path.join("C:/Users/Frederik/Masterarbeit/Data/Test", f))
    else:
        shutil.copy(os.path.join(gs_path, f), os.path.join("C:/Users/Frederik/Masterarbeit/Data/Train", f))
print("Done")
