import torch
import torch.nn as nn
import torch.nn.functional as F

from RocketleagueModel import RocketLeagueModel
from boolModel import BoolModel
from vec3model import Vec3model


class Independent(RocketLeagueModel):
    def __init__(self, config):
        super(Independent, self).__init__(config)

        self.models = {
            "bpos": Vec3model(config, 100),
            "bvel": Vec3model(config, 100),
            "bangvel": Vec3model(config, 100),
            "cpos": Vec3model(config, 100),
            "cforward": Vec3model(config, 100),
            "cup": Vec3model(config, 100),
            "cvel": Vec3model(config, 100),
            "cangvel": Vec3model(config, 100),
            "conground": BoolModel(config, 100),
            "cballtouch": BoolModel(config, 100),
            "chasjump": BoolModel(config, 100),
            "chasflip": BoolModel(config, 100),
            "cisdemo": BoolModel(config, 100),
        }
        self.init_train_models()

    def format_prediction(self, y_predicted):
        bpos, bvel, bangvel, cpos, cforward, cup, cvel, cangvel, conground, cballtouch, chasjump, chasflip, cisdemo = y_predicted
        return bpos, bvel, bangvel, cpos, cforward, cup, cvel, cangvel, \
            conground, cballtouch, chasjump, chasflip, cisdemo

    def format_losses(self):
        ls = ()
        for name in self.models:
            ls = ls + (self.models[name].loss,)
        return ls

    def init_train_models(self):
        self.train = {
            "bpos": self.config.train_ball_pos,
            "bvel": self.config.train_ball_vel,
            "bangvel": self.config.train_ball_ang_vel,
            "cpos": self.config.train_car_pos,
            "cforward": self.config.train_car_forward,
            "cup": self.config.train_car_up,
            "cvel": self.config.train_car_vel,
            "cangvel": self.config.train_car_ang_vel,
            "conground": self.config.train_car_on_ground,
            "cballtouch": self.config.train_car_ball_touch,
            "chasjump": self.config.train_car_has_jump,
            "chasflip": self.config.train_car_has_flip,
            "cisdemo": self.config.train_car_is_demo,
        }
