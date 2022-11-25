import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import log
from metrics import Metrics


class RocketLeagueModel(nn.Module):
    def __init__(self, config):
        super(RocketLeagueModel, self).__init__()
        self.config = config
        self.reset_running_losses()
        self.models = {}
        self.optimizers = {}
        self.train = {}
        self.init_train_models()

    def init_optim(self):
        for name in self.models:
            model = self.models[name]
            self.optimizers[name] = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

    def optim_zero_grad(self):
        for name in self.models:
            optimizer = self.optimizers[name]
            optimizer.zero_grad()

    def optim_step(self):
        for name in self.models:
            if self.train[name]:
                optimizer = self.optimizers[name]
                optimizer.step()

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

    def forward(self, x):
        y = ()
        for model in self.models:
            y = y + (self.models[model].forward(x),)
        return self.format_prediction(y)

    def format_prediction(self, y_predicted):
        # Overwrite
        pass

    def format_losses(self):
        # Overwrite
        pass

    def init_train_models(self):
        # Overwrite
        pass

    def accumulate_loss(self, b_pos_loss, b_vel_loss, b_ang_vel_loss, c_pos_loss, c_forward_loss, c_up_loss, c_vel_loss,
                        c_ang_vel_loss, c_on_ground_loss, c_ball_touch_loss, c_has_jump_loss, c_has_flip_loss,
                        c_is_demo_loss):
        loss = b_pos_loss + b_vel_loss + b_ang_vel_loss + c_pos_loss + c_forward_loss + c_up_loss + c_vel_loss + c_ang_vel_loss + c_on_ground_loss + c_ball_touch_loss + c_has_jump_loss + c_has_flip_loss + c_is_demo_loss
        return loss

    def update_training_steps(self, delta_steps):
        for name in self.models.keys():
            if self.train[name]:
                self.models[name].steps += delta_steps

    def to(self, device):
        for model in self.models:
            self.models[model].to(device)

    def load_checkpoint(self):
        start_epoch = 0
        steps = 0
        if not os.path.exists(self.config.checkpoint_path):
            os.mkdir(self.config.checkpoint_path)
        else:
            checkpoints = os.listdir(self.config.checkpoint_path)
            paths = [os.path.join(self.config.checkpoint_path, basename) for basename in checkpoints]
            if self.config.continue_from_checkpoint and len(checkpoints) > 0:
                cp_file = max(paths, key=os.path.getctime)
                log(f"Found existing model with that name, continue from latest checkpoint {cp_file}")
                loaded_checkpoint = torch.load(cp_file, map_location="cuda:0")
                start_epoch = loaded_checkpoint["epoch"]
                steps = loaded_checkpoint["step"]
                device = loaded_checkpoint["device"]
                for name in self.models.keys():
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
                        model.load_state_dict(loaded_checkpoint["model_state"])
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
        for name in self.models.keys():
            model = self.models[name]
            optimizer = self.optimizers[name]
            model_cp_path = os.path.join(self.config.checkpoint_path, name)
            checkpoint = {
                "step": model.steps,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict()
            }
            if not os.path.exists(model_cp_path):
                os.mkdir(model_cp_path)
            file_name = f"cp_{name}_{model.steps}.cp"
            torch.save(checkpoint, os.path.join(model_cp_path, file_name))


    def criterion(self, y_predicted, y_train, x_train):
        b_pos_pred, b_vel_pred, b_ang_vel_pred, c_pos_pred, c_forward_pred, c_up_pred, c_vel_pred, c_ang_vel_pred, c_on_ground_pred, c_ball_touch_pred, c_has_jump_pred, c_has_flip_pred, c_is_demo_pred = y_predicted

        b_pos_l, b_vel_l, b_ang_vel_l, c_pos_l, c_forward_l, c_up_l, c_vel_l, c_ang_vel_l, c_on_ground_l, c_ball_touch_l, c_has_jump_l, c_has_flip_l, c_is_demo_l = self.format_losses()
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        b_pos_loss = b_pos_l(b_pos_pred, y_train[:, 0:3])
        b_vel_loss = b_vel_l(b_vel_pred, y_train[:, 3:6])
        b_ang_vel_loss = b_ang_vel_l(b_ang_vel_pred, y_train[:, 6:9])

        c_pos_loss = c_pos_l(c_pos_pred, y_train[:, 9:12])
        c_forward_loss = c_forward_l(c_forward_pred, y_train[:, 12:15])
        c_up_loss = c_up_l(c_up_pred, y_train[:, 15:18])
        c_vel_loss = c_vel_l(c_vel_pred, y_train[:, 18:21])
        c_ang_vel_loss = c_ang_vel_l(c_ang_vel_pred, y_train[:, 21:24])

        n = y_train.shape[0]
        c_on_ground_loss = c_on_ground_l(c_on_ground_pred, y_train[:, 24].view(n, 1))
        c_ball_touch_loss = c_ball_touch_l(c_ball_touch_pred, y_train[:, 25].view(n, 1))
        c_has_jump_loss = c_has_jump_l(c_has_jump_pred, y_train[:, 26].view(n, 1))
        c_has_flip_loss = c_has_flip_l(c_has_flip_pred, y_train[:, 27].view(n, 1))
        c_is_demo_loss = c_is_demo_l(c_is_demo_pred, y_train[:, 28].view(n, 1))

        loss = self.accumulate_loss(b_pos_loss, b_vel_loss, b_ang_vel_loss, c_pos_loss, c_forward_loss, c_up_loss,
                                    c_vel_loss, c_ang_vel_loss, c_on_ground_loss, c_ball_touch_loss, c_has_jump_loss,
                                    c_has_flip_loss, c_is_demo_loss)

        self.running_loss += loss.item()
        self.running_b_pos_loss += b_pos_loss.item()
        self.running_b_vel_loss += b_vel_loss.item()
        self.running_b_ang_vel_loss += b_ang_vel_loss.item()
        self.running_c_pos_loss += c_pos_loss.item()
        self.running_c_forward_loss += c_forward_loss.item()
        self.running_c_up_loss += c_up_loss.item()
        self.running_c_vel_loss += c_vel_loss.item()
        self.running_c_ang_vel_loss += c_ang_vel_loss.item()
        self.running_c_on_ground_loss += c_on_ground_loss.item()
        self.running_c_ball_touch_loss += c_ball_touch_loss.item()
        self.running_c_has_jump_loss += c_has_jump_loss.item()
        self.running_c_has_flip_loss += c_has_flip_loss.item()
        self.running_c_is_demo_loss += c_is_demo_loss.item()

        M = Metrics(self.config)
        self.running_b_pos_diff += M.euclid(b_pos_pred, y_train[:, 0:3])
        self.running_b_vel_diff += M.euclid(b_vel_pred, y_train[:, 3:6])
        self.running_b_ang_vel_diff += M.euclid(b_ang_vel_pred, y_train[:, 6:9])
        self.running_c_pos_diff += M.euclid(c_pos_pred, y_train[:, 9:12])
        if self.config.delta_targets:
            self.running_c_forward_sim += M.cos_sim(c_forward_pred + x_train[:, 12:15],
                                                    y_train[:, 12:15] + x_train[:, 12:15])
            self.running_c_up_sim += M.cos_sim(c_up_pred + x_train[:, 15:18],
                                               y_train[:, 15:18] + x_train[:, 15:18])
        else:
            self.running_c_forward_sim += M.cos_sim(c_forward_pred, y_train[:, 12:15])
            self.running_c_up_sim += M.cos_sim(c_up_pred, y_train[:, 15:18])
        self.running_c_vel_diff += M.euclid(c_vel_pred, y_train[:, 18:21])
        self.running_c_ang_vel_diff += M.euclid(c_ang_vel_pred, y_train[:, 21:24])
        self.running_c_on_ground_acc += M.acc(c_on_ground_pred, y_train[:, 24].view(n, 1))
        self.running_c_ball_touch_acc += M.acc(c_ball_touch_pred, y_train[:, 25].view(n, 1))
        self.running_c_has_jump_acc += M.acc(c_has_jump_pred, y_train[:, 26].view(n, 1))
        self.running_c_has_flip_acc += M.acc(c_has_flip_pred, y_train[:, 27].view(n, 1))
        self.running_c_is_demo_acc += M.acc(c_is_demo_pred, y_train[:, 28].view(n, 1))

        return loss

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

        self.reset_running_losses()
