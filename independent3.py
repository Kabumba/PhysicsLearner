from BallVecModel import BallVecModel
from CarBoolModel import CarBoolModel
from CarRotModel import CarRotModel
from CarVecModel import CarVecModel
from RocketleagueModel import RocketLeagueModel
from input_formater import ball_input, car_input


class Independent3(RocketLeagueModel):
    def __init__(self, config):
        super(Independent3, self).__init__(config)

        self.models = {
            "bpos": BallVecModel(),
            "bvel": BallVecModel(),
            "bangvel": BallVecModel(),
            "cpos": CarVecModel(),
            "crot": CarRotModel(),
            "cvel": CarVecModel(),
            "cangvel": CarVecModel(),
            "conground": CarBoolModel(),
            "cballtouch": CarBoolModel(),
            "chasjump": CarBoolModel(),
            "chasflip": CarBoolModel(),
            "cisdemo": CarBoolModel(),
        }
        self.init_train_models()

    def forward(self, x):
        y_pred = (self.models["bpos"].forward(ball_input(x)),
              self.models["bvel"].forward(ball_input(x)),
              self.models["bangvel"].forward(ball_input(x)),
              self.models["cpos"].forward(car_input(x)),
              self.models["crot"].forward(car_input(x)),
              self.models["cvel"].forward(car_input(x)),
              self.models["cangvel"].forward(car_input(x)),
              self.models["conground"].forward(car_input(x)),
              self.models["cballtouch"].forward(car_input(x)),
              self.models["chasjump"].forward(car_input(x)),
              self.models["chasflip"].forward(car_input(x)),
              self.models["cisdemo"].forward(car_input(x)),
              )
        return self.format_prediction(y_pred)

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
