import argparse
import os
import sys

from configuration import Configuration
from logger import log
from trainer import start_training

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start or continue training with configurations')
    parser.add_argument('-n', '--name', help='Name of the experiment. Directory with this name should contain a '
                                             'configuration.yaml file where the hyperparameters for training are stored.')
    args = parser.parse_args()
    if args.name == "LiDo":
        configurations = [Configuration(name) for name in os.listdir("../Experimente")]
    else:
        configurations = [Configuration(args.name)]
    start_training(configurations)
    log("Done!")
    sys.exit()
