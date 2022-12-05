import os
import shutil
from IO_manager import Directories


dirs = Directories()
state_dir = dirs.BASE_DIR + "/Train"
trash_dir = dirs.BASE_DIR + "/Unused"
gs = os.listdir(state_dir)
upsi_files = []
for f in gs:
    base_name = os.path.splitext(os.path.basename(f))[0]
    if f.startswith("[GaussSteps"):
        shutil.move(os.path.join(state_dir, f), os.path.join(trash_dir, f))
    if f.startswith("[ConstantSplit,0."):
        shutil.move(os.path.join(state_dir, f), os.path.join(trash_dir, f))
    if f.startswith("[Interpolator"):
        shutil.move(os.path.join(state_dir, f), os.path.join(trash_dir, f))
print(f"done {len(upsi_files)}")