import os

import torch

from BallVecModel import BallVecModel
from CarBoolModel import CarBoolModel
from CarVecModel import CarVecModel
from RocketleagueModel import RocketLeagueModel
from VecModel import VecModel
from input_formater import ball_input, car_input
from logger import log
from metrics import Metrics
from output_formater import ball_output, car_output


class IndependentTest(RocketLeagueModel):
    def __init__(self, config):
        super(IndependentTest, self).__init__(config)

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

    def reset_running_losses(self):
        self.running_loss = 0
        self.running_b_pos_loss = 0
        self.running_b_vel_loss = 0
        self.running_b_ang_vel_loss = 0
        self.running_c_pos_loss = 0
        self.running_c_forward_loss = 0
        self.running_c_up_loss = 0
        self.running_c_vel_loss = 0
        self.running_c_ang_vel_loss = 0
        self.running_c_on_ground_loss = 0
        self.running_c_ball_touch_loss = 0
        self.running_c_has_jump_loss = 0
        self.running_c_has_flip_loss = 0
        self.running_c_is_demo_loss = 0

        self.running_b_pos_diff = 0
        self.running_b_vel_diff = 0
        self.running_b_ang_vel_diff = 0
        self.running_c_pos_diff = 0
        self.running_c_forward_sim = 0
        self.running_c_up_sim = 0
        self.running_c_vel_diff = 0
        self.running_c_ang_vel_diff = 0
        self.running_c_on_ground_acc = 0
        self.running_c_ball_touch_acc = 0
        self.running_c_has_jump_acc = 0
        self.running_c_has_flip_acc = 0
        self.running_c_is_demo_acc = 0

        self.running_b_pos_diff_in = 0
        self.running_b_vel_diff_in = 0
        self.running_b_ang_vel_diff_in = 0
        self.running_c_pos_diff_in = 0
        self.running_c_forward_sim_in = 0
        self.running_c_up_sim_in = 0
        self.running_c_vel_diff_in = 0
        self.running_c_ang_vel_diff_in = 0
        self.running_c_on_ground_acc_in = 0
        self.running_c_ball_touch_acc_in = 0
        self.running_c_has_jump_acc_in = 0
        self.running_c_has_flip_acc_in = 0
        self.running_c_is_demo_acc_in = 0

    def load_checkpoint(self, i=None):
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
                if i is None:
                    cph_file = max(cph_paths, key=os.path.getctime)
                    log(f"Found existing model with that name, continue from latest checkpoint {cph_file}")
                else:
                    cph_file = os.path.join(self.config.checkpoint_path, "cph_" + str(i) + ".cph")
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
                        if i is None:
                            cp_file = max(paths, key=os.path.getctime)
                            log(f"Found existing model for {name}, continue from latest checkpoint {cp_file}")
                        else:
                            cp_file = None
                            ending = str(i) + ".cp"
                            for f in paths:
                                if f.endswith(ending):
                                    cp_file = f
                            if cp_file is None:
                                raise
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

    def criterion(self, y_predicted, y_train, x_train):
        b_pos_pred, b_vel_pred, b_ang_vel_pred, c_pos_pred, c_forward_pred, c_up_pred, c_vel_pred, c_ang_vel_pred, c_on_ground_pred, c_ball_touch_pred, c_has_jump_pred, c_has_flip_pred, c_is_demo_pred = y_predicted
        # predictions = y_predicted

        b_pos_l, b_vel_l, b_ang_vel_l, c_pos_l, c_forward_l, c_up_l, c_vel_l, c_ang_vel_l, c_on_ground_l, c_ball_touch_l, c_has_jump_l, c_has_flip_l, c_is_demo_l = self.format_losses()
        # loss_functions = self.format_losses()

        M = Metrics(self.config)
        n = y_train.shape[0]
        ball_y = ball_output(x_train, y_train)
        car_y = car_output(x_train, y_train)
        zeros = torch.zeros(n, device=y_train.device)
        zero3 = torch.zeros((n,3), device=y_train.device)
        if self.config.train_ball_pos:
            b_pos_loss = b_pos_l(b_pos_pred, ball_y[:, 0:3])
            self.running_b_pos_loss += torch.mean(b_pos_loss).item()
            self.running_b_pos_diff += M.euclid(b_pos_pred, ball_y[:, 0:3], self.config.ball_pos_norm_factor)
            self.running_b_pos_diff_in += M.euclid(b_pos_pred, zero3, self.config.ball_pos_norm_factor)
        else:
            b_pos_loss = zeros

        if self.config.train_ball_vel:
            b_vel_loss = b_vel_l(b_vel_pred, ball_y[:, 3:6])
            self.running_b_vel_loss += torch.mean(b_vel_loss).item()
            self.running_b_vel_diff += M.euclid(b_vel_pred, ball_y[:, 3:6], self.config.ball_vel_norm_factor)
            self.running_b_vel_diff_in += M.euclid(b_vel_pred, zero3, self.config.ball_vel_norm_factor)
        else:
            b_vel_loss = zeros

        if self.config.train_ball_ang_vel:
            b_ang_vel_loss = b_ang_vel_l(b_ang_vel_pred, ball_y[:, 6:9])
            self.running_b_ang_vel_loss += torch.mean(b_ang_vel_loss).item()
            self.running_b_ang_vel_diff += M.euclid(b_ang_vel_pred, ball_y[:, 6:9],
                                                    self.config.ball_ang_vel_norm_factor)
            self.running_b_ang_vel_diff_in += M.euclid(b_ang_vel_pred, zero3,
                                                       self.config.ball_ang_vel_norm_factor)
        else:
            b_ang_vel_loss = zeros

        if self.config.train_car_pos:
            c_pos_loss = c_pos_l(c_pos_pred, car_y[:, 0:3])
            self.running_c_pos_loss += torch.mean(c_pos_loss).item()
            self.running_c_pos_diff += M.euclid(c_pos_pred, car_y[:, 0:3], self.config.car_pos_norm_factor)
            self.running_c_pos_diff_in += M.euclid(c_pos_pred, zero3, self.config.car_pos_norm_factor)
        else:
            c_pos_loss = zeros

        if self.config.train_car_forward:
            c_forward_loss = c_forward_l(c_forward_pred, car_y[:, 3:6])
            self.running_c_forward_loss += torch.mean(c_forward_loss).item()
            if isinstance(c_forward_pred, tuple):
                c_forward_pred = torch.cat(c_forward_pred, dim=1)
            self.running_c_forward_sim += M.cos_sim(c_forward_pred + x_train[:, 12:15],
                                                    car_y[:, 3:6] + x_train[:, 12:15])
            self.running_c_forward_sim_in += M.cos_sim(c_forward_pred + x_train[:, 12:15], x_train[:, 12:15])
        else:
            c_forward_loss = zeros

        if self.config.train_car_up:
            c_up_loss = c_up_l(c_up_pred, car_y[:, 6:9])
            self.running_c_up_loss += torch.mean(c_up_loss).item()
            if isinstance(c_up_pred, tuple):
                c_up_pred = torch.cat(c_up_pred, dim=1)
            self.running_c_up_sim += M.cos_sim(c_up_pred + x_train[:, 15:18], car_y[:, 6:9] + x_train[:, 15:18])
            self.running_c_up_sim_in += M.cos_sim(c_forward_pred + x_train[:, 12:15], x_train[:, 12:15])
        else:
            c_up_loss = zeros

        if self.config.train_car_vel:
            c_vel_loss = c_vel_l(c_vel_pred, car_y[:, 9:12])
            self.running_c_vel_loss += torch.mean(c_vel_loss).item()
            self.running_c_vel_diff += M.euclid(c_vel_pred, car_y[:, 9:12], self.config.car_vel_norm_factor)
            self.running_c_vel_diff_in += M.euclid(c_vel_pred, zero3, self.config.car_vel_norm_factor)
        else:
            c_vel_loss = zeros

        if self.config.train_car_ang_vel:
            c_ang_vel_loss = c_ang_vel_l(c_ang_vel_pred, car_y[:, 12:15])
            self.running_c_ang_vel_loss += torch.mean(c_ang_vel_loss).item()
            self.running_c_ang_vel_diff += M.euclid(c_ang_vel_pred, car_y[:, 12:15],
                                                    self.config.car_ang_vel_norm_factor)
            self.running_c_ang_vel_diff_in += M.euclid(c_ang_vel_pred, zero3, self.config.car_ang_vel_norm_factor)
        else:
            c_ang_vel_loss = zeros

        if self.config.train_car_on_ground:
            c_on_ground_loss = c_on_ground_l(c_on_ground_pred, car_y[:, 15].view(n, 1))
            self.running_c_on_ground_loss += torch.mean(c_on_ground_loss).item()
            self.running_c_on_ground_acc += M.acc(c_on_ground_pred, car_y[:, 15].view(n, 1))
            self.running_c_on_ground_acc_in += M.acc(c_on_ground_pred, x_train[:, 25].view(n, 1))
        else:
            c_on_ground_loss = zeros

        if self.config.train_car_ball_touch:
            c_ball_touch_loss = c_ball_touch_l(c_ball_touch_pred, car_y[:, 16].view(n, 1))
            self.running_c_ball_touch_loss += torch.mean(c_ball_touch_loss).item()
            self.running_c_ball_touch_acc += M.acc(c_ball_touch_pred, car_y[:, 16].view(n, 1))
            self.running_c_ball_touch_acc_in += M.acc(c_on_ground_pred, x_train[:, 26].view(n, 1))
        else:
            c_ball_touch_loss = zeros

        if self.config.train_car_has_jump:
            c_has_jump_loss = c_has_jump_l(c_has_jump_pred, car_y[:, 17].view(n, 1))
            self.running_c_has_jump_loss += torch.mean(c_has_jump_loss).item()
            self.running_c_has_jump_acc += M.acc(c_has_jump_pred, car_y[:, 17].view(n, 1))
            self.running_c_has_jump_acc_in += M.acc(c_has_jump_pred, x_train[:, 27].view(n, 1))
        else:
            c_has_jump_loss = zeros

        if self.config.train_car_has_flip:
            c_has_flip_loss = c_has_flip_l(c_has_flip_pred, car_y[:, 18].view(n, 1))
            self.running_c_has_flip_loss += torch.mean(c_has_flip_loss).item()
            self.running_c_has_flip_acc += M.acc(c_has_flip_pred, car_y[:, 18].view(n, 1))
            self.running_c_has_flip_acc_in += M.acc(c_has_flip_pred, x_train[:, 28].view(n, 1))
        else:
            c_has_flip_loss = zeros

        if self.config.train_car_is_demo:
            c_is_demo_loss = c_is_demo_l(c_is_demo_pred, car_y[:, 19].view(n, 1))
            self.running_c_is_demo_loss += torch.mean(c_is_demo_loss).item()
            self.running_c_is_demo_acc += M.acc(c_is_demo_pred, car_y[:, 19].view(n, 1))
            self.running_c_is_demo_acc_in += M.acc(c_is_demo_pred, x_train[:, 29].view(n, 1))
        else:
            c_is_demo_loss = zeros

        losses = self.accumulate_loss(b_pos_loss, b_vel_loss, b_ang_vel_loss, c_pos_loss, c_forward_loss, c_up_loss,
                                      c_vel_loss, c_ang_vel_loss, c_on_ground_loss, c_ball_touch_loss, c_has_jump_loss,
                                      c_has_flip_loss, c_is_demo_loss)
        feedback_batch = None
        if self.config.loss_feedback > 0:
            indices = torch.argsort(losses, descending=True)[:self.config.loss_feedback]
            feedback_batch = x_train[indices, :], y_train[indices, :]
        loss = torch.mean(losses)

        self.running_loss += loss.item()

        return loss, feedback_batch

    def tensor_log(self, writer, iterations_last_tensor_log, global_step):
        f = 1 / iterations_last_tensor_log
        writer.add_scalar('00 training loss', self.running_loss * f, global_step)
        writer.add_scalar('01 ball position loss', self.running_b_pos_loss * f, global_step)
        writer.add_scalar('02 ball velocity loss', self.running_b_vel_loss * f, global_step)
        writer.add_scalar('03 ball angular velocity loss', self.running_b_ang_vel_loss * f, global_step)
        writer.add_scalar('04 car position loss', self.running_c_pos_loss * f, global_step)
        writer.add_scalar('05 car forward loss', self.running_c_forward_loss * f, global_step)
        writer.add_scalar('06 car up loss', self.running_c_up_loss * f, global_step)
        writer.add_scalar('07 car velocity loss', self.running_c_vel_loss * f, global_step)
        writer.add_scalar('08 car angular velocity loss', self.running_c_ang_vel_loss * f, global_step)
        writer.add_scalar('09 car on ground loss', self.running_c_on_ground_loss * f, global_step)
        writer.add_scalar('10 car ball touch loss', self.running_c_ball_touch_loss * f, global_step)
        writer.add_scalar('11 car has jump loss', self.running_c_has_jump_loss * f, global_step)
        writer.add_scalar('12 car has flip loss', self.running_c_has_flip_loss * f, global_step)
        writer.add_scalar('13 car is demo loss', self.running_c_is_demo_loss * f, global_step)
        writer.add_scalar('01 ball position diff', self.running_b_pos_diff * f, global_step)
        writer.add_scalar('02 ball velocity diff', self.running_b_vel_diff * f, global_step)
        writer.add_scalar('03 ball angular velocity diff', self.running_b_ang_vel_diff * f, global_step)
        writer.add_scalar('04 car position diff', self.running_c_pos_diff * f, global_step)
        writer.add_scalar('05 car forward similarity', self.running_c_forward_sim * f, global_step)
        writer.add_scalar('06 car up similarity', self.running_c_up_sim * f, global_step)
        writer.add_scalar('07 car velocity diff', self.running_c_vel_diff * f, global_step)
        writer.add_scalar('08 car angular velocity diff', self.running_c_ang_vel_diff * f, global_step)
        writer.add_scalar('09 car on ground accuracy', self.running_c_on_ground_acc * f, global_step)
        writer.add_scalar('10 car ball touch accuracy', self.running_c_ball_touch_acc * f, global_step)
        writer.add_scalar('11 car has jump accuracy', self.running_c_has_jump_acc * f, global_step)
        writer.add_scalar('12 car has flip accuracy', self.running_c_has_flip_acc * f, global_step)
        writer.add_scalar('13 car is demo accuracy', self.running_c_is_demo_acc * f, global_step)

        writer.add_scalar('01 ball position diff in', self.running_b_pos_diff_in * f, global_step)
        writer.add_scalar('02 ball velocity diff in', self.running_b_vel_diff_in * f, global_step)
        writer.add_scalar('03 ball angular velocity diff in', self.running_b_ang_vel_diff_in * f, global_step)
        writer.add_scalar('04 car position diff in', self.running_c_pos_diff_in * f, global_step)
        writer.add_scalar('05 car forward similarity in', self.running_c_forward_sim_in * f, global_step)
        writer.add_scalar('06 car up similarity in', self.running_c_up_sim_in * f, global_step)
        writer.add_scalar('07 car velocity diff in', self.running_c_vel_diff_in * f, global_step)
        writer.add_scalar('08 car angular velocity diff in', self.running_c_ang_vel_diff_in * f, global_step)
        writer.add_scalar('09 car on ground accuracy in', self.running_c_on_ground_acc_in * f, global_step)
        writer.add_scalar('10 car ball touch accuracy in', self.running_c_ball_touch_acc_in * f, global_step)
        writer.add_scalar('11 car has jump accuracy in', self.running_c_has_jump_acc_in * f, global_step)
        writer.add_scalar('12 car has flip accuracy in', self.running_c_has_flip_acc_in * f, global_step)
        writer.add_scalar('13 car is demo accuracy in', self.running_c_is_demo_acc_in * f, global_step)

        self.reset_running_losses()
