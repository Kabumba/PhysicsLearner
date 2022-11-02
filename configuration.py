import yaml

from logger import log


class Configuration:
    def __init__(self, name):
        self.name = name
        self.base_path = ".."
        self.result_path = self.base_path + "/Experimente/" + name
        self.configuration_path = self.result_path + "/configuration.yaml"
        self.data_path = self.base_path + "/Data"
        self.train_path = self.data_path + "/Train"
        self.test_path = self.data_path + "/Test"
        self.batch_size = 1024
        self.learning_rate = 0.003
        self.num_epoch = 3
        self.num_car_in = 2
        self.num_car_out = 2
        self.mirror_and_invert = True
        self.invert = True
        self.mirror = True
        self.ball_out = True
        self.ball_in = True
        self.setup()

    def setup(self):
        with open(self.configuration_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                for c in config:
                    for p in config[c]:
                        self.__dict__[p] = config[c][p]
            except yaml.YAMLError as exc:
                log(exc)
