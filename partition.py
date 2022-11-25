import math
import os
import random


def create_partition(config, num_partitions):
    kickoff_files = os.listdir(config.train_path)
    partitions = []
    for i in range(num_partitions):
        partitions.append([])
    for f in kickoff_files:
        if f.endswith(".pt"):
            r = math.floor(random.Random().uniform(0, num_partitions))
            partitions[r].append(f)
    return partitions