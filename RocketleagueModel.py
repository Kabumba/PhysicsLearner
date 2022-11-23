import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import Metrics


class RocketLeagueModel(nn.Module):
    def __init__(self, config):
        super(RocketLeagueModel, self).__init__()
        self.config = config
        self.reset_running_losses()

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
        # Overwrite
        pass

    def format_prediction(self, y_predicted):
        # Overwrite
        pass

    def criterion(self, y_predicted, y_train, x_train):
        b_pos_pred, b_vel_pred, b_ang_vel_pred, c_pos_pred, c_forward_pred, c_up_pred, c_vel_pred, c_ang_vel_pred, c_on_ground_pred, c_ball_touch_pred, c_has_jump_pred, c_has_flip_pred, c_is_demo_pred = self.format_prediction(
            y_predicted)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        b_pos_loss = mse_loss(b_pos_pred, y_train[:, 0:3])
        b_vel_loss = mse_loss(b_vel_pred, y_train[:, 3:6])
        b_ang_vel_loss = mse_loss(b_ang_vel_pred, y_train[:, 6:9])

        c_pos_loss = mse_loss(c_pos_pred, y_train[:, 9:12])
        c_forward_loss = mse_loss(c_forward_pred, y_train[:, 12:15])
        c_up_loss = mse_loss(c_up_pred, y_train[:, 15:18])
        c_vel_loss = mse_loss(c_vel_pred, y_train[:, 18:21])
        c_ang_vel_loss = mse_loss(c_ang_vel_pred, y_train[:, 21:24])

        c_on_ground_loss = bce_loss(c_on_ground_pred, y_train[:, 24])
        c_ball_touch_loss = bce_loss(c_ball_touch_pred, y_train[:, 25])
        c_has_jump_loss = bce_loss(c_has_jump_pred, y_train[:, 26])
        c_has_flip_loss = bce_loss(c_has_flip_pred, y_train[:, 27])
        c_is_demo_loss = bce_loss(c_is_demo_pred, y_train[:, 28])

        loss = self.accumulate_loss(b_pos_loss, b_vel_loss, b_ang_vel_loss, c_pos_loss, c_forward_loss, c_up_loss,
                                    c_vel_loss, c_ang_vel_loss, c_on_ground_loss, c_ball_touch_loss, c_has_jump_loss, c_has_flip_loss,
                                    c_is_demo_loss)

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
        self.running_c_on_ground_acc += M.acc(c_on_ground_pred, y_train[:, 24])
        self.running_c_ball_touch_acc += M.acc(c_ball_touch_pred, y_train[:, 25])
        self.running_c_has_jump_acc += M.acc(c_has_jump_pred, y_train[:, 26])
        self.running_c_has_flip_acc += M.acc(c_has_flip_pred, y_train[:, 27])
        self.running_c_is_demo_acc += M.acc(c_is_demo_pred, y_train[:, 28])

        return loss

    def accumulate_loss(self, b_pos_loss, b_vel_loss, b_ang_vel_loss, c_pos_loss, c_forward_loss, c_up_loss, c_vel_loss,
                        c_ang_vel_loss, c_on_ground_loss, c_ball_touch_loss, c_has_jump_loss, c_has_flip_loss, c_is_demo_loss):
        # Overwrite
        pass

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
