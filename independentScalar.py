import os

import torch

from BoolModel import BoolModel
from RocketleagueModel import RocketLeagueModel
from VecModel import VecModel
from input_formater import ball_input, car_input
from logger import log


class IndependentScalar(RocketLeagueModel):
    def __init__(self, config):
        super(IndependentScalar, self).__init__(config)

        self.models = {
            "bpos": VecModel(config, 43, config.hidden_size),
            "bvel": VecModel(config, 43, config.hidden_size),
            "bangvel": VecModel(config, 43, config.hidden_size),
            "cpos": VecModel(config, 71, config.hidden_size),
            "cforward": VecModel(config, 71, config.hidden_size),
            "cup": VecModel(config, 71, config.hidden_size),
            "cvel": VecModel(config, 71, config.hidden_size),
            "cangvel": VecModel(config, 71, config.hidden_size),
            "conground": BoolModel(config, 71, config.hidden_size),
            "cballtouch": BoolModel(config, 71, config.hidden_size),
            "chasjump": BoolModel(config, 71, config.hidden_size),
            "chasflip": BoolModel(config, 71, config.hidden_size),
            "cisdemo": BoolModel(config, 71, config.hidden_size),
        }
        self.init_train_models()

    def forward(self, x):
        y_pred = (self.models["bpos"].forward(ball_input(x)),
                  self.models["bvel"].forward(ball_input(x)),
                  self.models["bangvel"].forward(ball_input(x)),
                  self.models["cpos"].forward(car_input(x)),
                  self.models["cforward"].forward(car_input(x)),
                  self.models["cup"].forward(car_input(x)),
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
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
            if isinstance(model, VecModel):
                model_state = {}
                for n in model.models:
                    model_state[n] = model.models[n].state_dict()
                optimizer_state = ()
                for o in optimizer:
                    optimizer_state = optimizer_state + (o.state_dict(),)
            checkpoint = {
                "step": model.steps,
                "model_state": model_state,
                "optim_state": optimizer_state
            }
            if not os.path.exists(model_cp_path):
                os.mkdir(model_cp_path)
            file_name = f"cp_{name}_{model.steps}.cp"
            torch.save(checkpoint, os.path.join(model_cp_path, file_name))
