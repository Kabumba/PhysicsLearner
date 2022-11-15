import argparse

from configuration import Configuration
from logger import log
from trainer import start_training

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start or continue training with configurations')
    parser.add_argument('-n', '--name', help='Name of the experiment. Directory with this name should contain a '
                                             'configuration.yaml file where the hyperparameters for training are stored.')
    args = parser.parse_args()
    configuration = Configuration(args.name)
    start_training(configuration)
    log("Done!")
