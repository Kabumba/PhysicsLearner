import os

import torch

from BallVecModel import BallVecModel
from CarBoolModel import CarBoolModel
from CarVecModel import CarVecModel
from RocketleagueModel import RocketLeagueModel
from VecModel import VecModel
from input_formater import ball_input, car_input
from logger import log


class Independent4(RocketLeagueModel):
    def __init__(self, config):
        super(Independent4, self).__init__(config)

        self.models = {
            "bpos": BallVecModel(config),
            "bvel": VecModel(config, 43, config.hidden_size),
            "bangvel": BallVecModel(config),
            "cpos": CarVecModel(config),
            "cforward": CarVecModel(config),
            "cup": CarVecModel(config),
            "cvel": VecModel(config, 71, config.hidden_size),
            "cangvel": CarVecModel(config),
            "conground": CarBoolModel(config),
            "cballtouch": CarBoolModel(config),
            "chasjump": CarBoolModel(config),
            "chasflip": CarBoolModel(config),
            "cisdemo": CarBoolModel(config),
        }
        self.init_train_models()

    def forward(self, x):
        if self.config.train_ball_pos:
            y1 = self.models["bpos"].forward(ball_input(x))
        else:
            y1 = None

        if self.config.train_ball_vel:
            y2 = self.models["bvel"].forward(ball_input(x))
        else:
            y2 = None

        if self.config.train_ball_ang_vel:
            y3 = self.models["bangvel"].forward(ball_input(x))
        else:
            y3 = None

        if self.config.train_car_pos:
            y4 = self.models["cpos"].forward(car_input(x))
        else:
            y4 = None

        if self.config.train_car_forward:
            y5 = self.models["cforward"].forward(car_input(x))
        else:
            y5 = None

        if self.config.train_car_up:
            y6 = self.models["cup"].forward(car_input(x))
        else:
            y6 = None

        if self.config.train_car_vel:
            y7 = self.models["cvel"].forward(car_input(x))
        else:
            y7 = None

        if self.config.train_car_ang_vel:
            y8 = self.models["cangvel"].forward(car_input(x))
        else:
            y8 = None

        if self.config.train_car_on_ground:
            y9 = self.models["conground"].forward(car_input(x))
        else:
            y9 = None

        if self.config.train_car_ball_touch:
            y10 = self.models["cballtouch"].forward(car_input(x))
        else:
            y10 = None

        if self.config.train_car_has_jump:
            y11 = self.models["chasjump"].forward(car_input(x))
        else:
            y11 = None

        if self.config.train_car_has_flip:
            y12 = self.models["chasflip"].forward(car_input(x))
        else:
            y12 = None

        if self.config.train_car_is_demo:
            y13 = self.models["cisdemo"].forward(car_input(x))
        else:
            y13 = None

        y_pred = y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13
        return self.format_prediction(y_pred)

    def format_prediction(self, y_predicted):
        return y_predicted

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

    def load_checkpoint(self):
        start_epoch = 0
        steps = 0
        if not os.path.exists(self.config.checkpoint_path):
            os.mkdir(self.config.checkpoint_path)
        else:
            checkpoints = os.listdir(self.config.checkpoint_path)
            paths = [os.path.join(self.config.checkpoint_path, basename) for basename in checkpoints]
            cph_paths = []
            for p in paths:
                if p.endswith(".cph"):
                    cph_paths.append(p)
            if self.config.continue_from_checkpoint and len(checkpoints) > 0:
                cph_file = max(cph_paths, key=os.path.getctime)
                log(f"Found existing model with that name, continue from latest checkpoint {cph_file}")
                loaded_checkpoint = torch.load(cph_file, map_location="cuda:0")
                start_epoch = loaded_checkpoint["epoch"]
                steps = loaded_checkpoint["step"]
                device = loaded_checkpoint["device"]
                for name in self.models:
                    model = self.models[name]
                    optimizer = self.optimizers[name]
                    model_cp_path = os.path.join(self.config.checkpoint_path, name)
                    checkpoints = os.listdir(model_cp_path)
                    paths = [os.path.join(model_cp_path, basename) for basename in checkpoints]
                    if self.config.continue_from_checkpoint and len(checkpoints) > 0:
                        cp_file = max(paths, key=os.path.getctime)
                        log(f"Found existing model for {name}, continue from latest checkpoint {cp_file}")
                        loaded_checkpoint = torch.load(cp_file, map_location="cuda:0")
                        model.steps = loaded_checkpoint["step"]
                        if isinstance(model, VecModel):
                            for n in model.models:
                                model.models[n].load_state_dict(loaded_checkpoint["model_state"][n])
                            if self.config.load_optim:
                                for i in range(len(optimizer)):
                                    optimizer[i].load_state_dict(loaded_checkpoint["optim_state"][i])
                        else:
                            model.load_state_dict(loaded_checkpoint["model_state"])
                            if self.config.load_optim:
                                optimizer.load_state_dict(loaded_checkpoint["optim_state"])
        return start_epoch, steps

    def save_checkpoint(self, epoch, device, steps):
        checkpoint = {
            "epoch": epoch,
            "device": device,
            "step": steps,
        }
        file_name = "cph_" + str(steps) + ".cph"
        log(f"Saved Checkpoint after {steps} as {file_name}")
        torch.save(checkpoint, os.path.join(self.config.checkpoint_path, file_name))
        for name in self.models:
            model = self.models[name]
            optimizer = self.optimizers[name]
            model_cp_path = os.path.join(self.config.checkpoint_path, name)
            if isinstance(model, VecModel):
                model_state = {}
                for n in model.models:
                    model_state[n] = model.models[n].state_dict()
                optimizer_state = ()
                for o in optimizer:
                    optimizer_state = optimizer_state + (o.state_dict(),)
            else:
                model_state = model.state_dict()
                optimizer_state = optimizer.state_dict()
            checkpoint = {
                "step": model.steps,
                "model_state": model_state,
                "optim_state": optimizer_state
            }
            if not os.path.exists(model_cp_path):
                os.mkdir(model_cp_path)
            file_name = f"cp_{name}_{model.steps}.cp"
            torch.save(checkpoint, os.path.join(model_cp_path, file_name))
