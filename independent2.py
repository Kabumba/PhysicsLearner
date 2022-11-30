import torch
import torch.nn as nn
import torch.nn.functional as F

from Bool2Model import Bool2Model
from RocketleagueModel import RocketLeagueModel
from RotModel import RotModel
from Vec32Model import Vec32Model
from BoolModel import BoolModel
from vec3model import Vec3model


class Independent2(RocketLeagueModel):
    def __init__(self, config):
        super(Independent2, self).__init__(config)

        self.models = {
            "bpos": Vec32Model(config, 100),
            "bvel": Vec32Model(config, 100),
            "bangvel": Vec32Model(config, 100),
            "cpos": Vec32Model(config, 100),
            "crot": RotModel(config, 100),
            "cvel": Vec32Model(config, 100),
            "cangvel": Vec32Model(config, 100),
            "conground": Bool2Model(config, 100),
            "cballtouch": Bool2Model(config, 100),
            "chasjump": Bool2Model(config, 100),
            "chasflip": Bool2Model(config, 100),
            "cisdemo": Bool2Model(config, 100),
        }
        self.init_train_models()

    def format_prediction(self, y_predicted):
        bpos, bvel, bangvel, cpos, c_rot, cvel, cangvel, conground, cballtouch, chasjump, chasflip, cisdemo = y_predicted
        return bpos, bvel, bangvel, cpos, c_rot[:, 0:3], c_rot[:, 3:6], cvel, cangvel, \
            conground, cballtouch, chasjump, chasflip, cisdemo

    def format_losses(self):
        ls = (self.models["bpos"].loss,
              self.models["bvel"].loss,
              self.models["bangvel"].loss,
              self.models["cpos"].loss,
              self.models["crot"].loss,
              self.models["crot"].loss,
              self.models["cvel"].loss,
              self.models["cangvel"].loss,
              self.models["conground"].loss,
              self.models["cballtouch"].loss,
              self.models["chasjump"].loss,
              self.models["chasflip"].loss,
              self.models["cisdemo"].loss,
              )
        return ls

    def init_train_models(self):
        self.train = {
            "bpos": self.config.train_ball_pos,
            "bvel": self.config.train_ball_vel,
            "bangvel": self.config.train_ball_ang_vel,
            "cpos": self.config.train_car_pos,
            "crot": self.config.train_car_forward and self.config.train_car_up,
            "cvel": self.config.train_car_vel,
            "cangvel": self.config.train_car_ang_vel,
            "conground": self.config.train_car_on_ground,
            "cballtouch": self.config.train_car_ball_touch,
            "chasjump": self.config.train_car_has_jump,
            "chasflip": self.config.train_car_has_flip,
            "cisdemo": self.config.train_car_is_demo,
        }
