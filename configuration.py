import yaml

from logger import log


class Configuration:
    def __init__(self, name):
        self.name = name
        self.base_path = ".."
        self.result_path = self.base_path + "/Experimente/" + name
        self.tensorboard_path = self.result_path + "/Tensorboard"
        self.checkpoint_path = self.result_path + "/Checkpoints"
        self.model_path = self.result_path + "/model.pth"
        self.configuration_path = self.result_path + "/configuration.yaml"
        self.data_path = self.base_path + "/Data"
        self.train_path = self.data_path + "/Train"
        self.test_path = self.data_path + "/Test"
        self.batch_size = 64
        self.learning_rate = 0.003
        self.hidden_size = 200
        self.num_hidden_layers = 2
        self.num_epoch = 3
        self.epochs_per_checkpoint = 1
        self.iterations_per_log = 1000
        self.num_car_in = 2
        self.num_car_out = 2
        self.in_size = 0
        self.out_size = 0
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
        if self.ball_in:
            self.in_size += 9
        self.in_size += self.num_car_in * 46
        if self.ball_out:
            self.out_size += 9
        self.out_size += self.num_car_out * 20
