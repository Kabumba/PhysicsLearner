import torch
import torch.nn as nn
import torch.nn.functional as F

from RocketleagueModel import RocketLeagueModel
from metrics import Metrics



class Naive(nn.Module):
    def __init__(self, config):
        super(Naive, self).__init__()

        self.fc1 = nn.Linear(config.in_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, config.out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def criterion(self, y_predicted, y_train):
        return nn.MSELoss(y_predicted, y_train)


class Split(nn.Module):
    def __init__(self, config):
        super(Split, self).__init__()
        self.config = config
        shared_size = 200
        ball_size = 30
        car_size = 60

        self.fc_shared = nn.Linear(config.in_size, shared_size)
        self.fc_ball1 = nn.Linear(shared_size, ball_size)
        self.fc_car1 = nn.Linear(shared_size, car_size)
        self.fc_ball2 = nn.Linear(ball_size, 9)  # pos, vel, ang_vel
        self.fc_car2 = nn.Linear(car_size, 9)  # pos, vel, ang_vel
        self.fc_car_rot = nn.Linear(car_size, 6)  # forward, up
        self.fc_car_bool = nn.Linear(car_size, 5)
        """self.fc_on_ground = nn.Linear(car_size, 1) 
        self.fc_ball_touch = nn.Linear(car_size, 1) 
        self.fc_has_jump = nn.Linear(car_size, 1)
        self.fc_has_flip = nn.Linear(car_size, 1)
        self.fc_is_demo = nn.Linear(car_size, 1)"""
        self.reset_running_losses()

    def reset_running_losses(self):
        self.running_loss = 0
        self.rb_pos_loss = 0
        self.rb_vel_loss = 0
        self.rb_ang_vel_loss = 0
        self.rc_pos_loss = 0
        self.rc_forward_loss = 0
        self.rc_up_loss = 0
        self.rc_vel_loss = 0
        self.rc_ang_vel_loss = 0
        self.rc_on_ground_loss = 0
        self.rc_ball_touch_loss = 0
        self.rc_has_jump_loss = 0
        self.rc_has_flip_loss = 0
        self.rc_is_demo_loss = 0
        self.rb_pos_diff = 0
        self.rb_vel_diff = 0
        self.rb_ang_vel_diff = 0
        self.rc_pos_diff = 0
        self.rc_forward_sim = 0
        self.rc_up_sim = 0
        self.rc_vel_diff = 0
        self.rc_ang_vel_diff = 0
        self.rc_on_ground_acc = 0
        self.rc_ball_touch_acc = 0
        self.rc_has_jump_acc = 0
        self.rc_has_flip_acc = 0
        self.rc_is_demo_acc = 0

    def forward(self, x):
        x = F.relu(self.fc_shared(x))
        xb = F.relu(self.fc_ball1(x))
        xb = self.fc_ball2(xb)
        xc = F.relu(self.fc_car1(x))
        xc1 = self.fc_car2(xc)
        xc_rot = torch.sin(self.fc_car_rot(xc))
        xc_bool = torch.sigmoid((self.fc_car_bool(xc)))

        """xb += x[:, 0:9]
        xc[:, 0:3] += x[:, 9:12]
        xc[:, 3:9] += x[:, 18:24]"""

        """x_c_on_ground = torch.sigmoid(self.fc_on_ground(x_c))
        x_c_ball_touch = torch.sigmoid(self.fc_ball_touch(x_c))
        x_c_has_jump = torch.sigmoid(self.fc_has_jump(x_c))
        x_c_has_flip = torch.sigmoid(self.fc_has_flip(x_c))
        x_c_is_demo = torch.sigmoid(self.fc_is_demo(x_c))"""
        return xb, xc1, xc_rot, xc_bool  # x_c_on_ground, x_c_ball_touch, x_c_has_jump, x_c_has_flip, x_c_is_demo

    def criterion(self, y_predicted, y_train):
        # yp_b, yp_c, yp_rot, yp_on_ground, yp_ball_touch, yp_has_jump, yp_has_flip, yp_is_demo = y_predicted
        yp_b, yp_c, yp_rot, yp_bool = y_predicted
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        b_pos_loss = mse_loss(yp_b[:, 0:3], y_train[:, 0:3])
        b_vel_loss = mse_loss(yp_b[:, 3:6], y_train[:, 3:6])
        b_ang_vel_loss = mse_loss(yp_b[:, 6:9], y_train[:, 6:9])

        c_pos_loss = mse_loss(yp_c[:, 0:3], y_train[:, 9:12])
        c_forward_loss = mse_loss(yp_rot[:, 0:3], y_train[:, 12:15])
        c_up_loss = mse_loss(yp_rot[:, 3:6], y_train[:, 15:18])
        c_vel_loss = mse_loss(yp_c[:, 3:6], y_train[:, 18:21])
        c_ang_vel_loss = mse_loss(yp_c[:, 6:9], y_train[:, 21:24])

        mse = (
                      b_pos_loss + b_vel_loss + b_ang_vel_loss + c_pos_loss + c_forward_loss + c_up_loss + c_vel_loss + c_ang_vel_loss) / 8.0

        c_on_ground_loss = bce_loss(yp_bool[:, 0], y_train[:, 24])
        c_ball_touch_loss = bce_loss(yp_bool[:, 1], y_train[:, 25])
        c_has_jump_loss = bce_loss(yp_bool[:, 2], y_train[:, 26])
        c_has_flip_loss = bce_loss(yp_bool[:, 3], y_train[:, 27])
        c_is_demo_loss = bce_loss(yp_bool[:, 4], y_train[:, 28])

        bce = c_on_ground_loss + c_ball_touch_loss + c_has_jump_loss + c_has_flip_loss + c_is_demo_loss

        loss = mse + bce

        self.running_loss += loss.item()
        self.rb_pos_loss += b_pos_loss.item()
        self.rb_vel_loss += b_vel_loss.item()
        self.rb_ang_vel_loss += b_ang_vel_loss.item()
        self.rc_pos_loss += c_pos_loss.item()
        self.rc_forward_loss += c_forward_loss.item()
        self.rc_up_loss += c_up_loss.item()
        self.rc_vel_loss += c_vel_loss.item()
        self.rc_ang_vel_loss += c_ang_vel_loss.item()
        self.rc_on_ground_loss += c_on_ground_loss.item()
        self.rc_ball_touch_loss += c_ball_touch_loss.item()
        self.rc_has_jump_loss += c_has_jump_loss.item()
        self.rc_has_flip_loss += c_has_flip_loss.item()
        self.rc_is_demo_loss += c_is_demo_loss.item()

        M = Metrics(self.config)
        self.rb_pos_diff += M.euclid(yp_b[:, 0:3], y_train[:, 0:3])
        self.rb_vel_diff += M.euclid(yp_b[:, 3:6], y_train[:, 3:6])
        self.rb_ang_vel_diff += M.euclid(yp_b[:, 6:9], y_train[:, 6:9])
        self.rc_pos_diff += M.euclid(yp_c[:, 0:3], y_train[:, 9:12])
        self.rc_forward_sim += M.cos_sim(yp_rot[:, 0:3], y_train[:, 12:15])
        self.rc_up_sim += M.cos_sim(yp_rot[:, 3:6], y_train[:, 15:18])
        self.rc_vel_diff += M.euclid(yp_c[:, 3:6], y_train[:, 18:21])
        self.rc_ang_vel_diff += M.euclid(yp_c[:, 6:9], y_train[:, 21:24])
        self.rc_on_ground_acc += M.acc(yp_bool[:, 0], y_train[:, 24])
        self.rc_ball_touch_acc += M.acc(yp_bool[:, 1], y_train[:, 25])
        self.rc_has_jump_acc += M.acc(yp_bool[:, 2], y_train[:, 26])
        self.rc_has_flip_acc += M.acc(yp_bool[:, 3], y_train[:, 27])
        self.rc_is_demo_acc += M.acc(yp_bool[:, 4], y_train[:, 28])

        return loss

    def tensor_log(self, writer, iterations_last_tensor_log, global_step):
        f = 1 / iterations_last_tensor_log
        writer.add_scalar('00 training loss', self.running_loss * f, global_step)
        writer.add_scalar('01 ball position loss', self.rb_pos_loss * f, global_step)
        writer.add_scalar('02 ball velocity loss', self.rb_vel_loss * f, global_step)
        writer.add_scalar('03 ball angular velocity loss', self.rb_ang_vel_loss * f, global_step)
        writer.add_scalar('04 car position loss', self.rc_pos_loss * f, global_step)
        writer.add_scalar('05 car forward loss', self.rc_forward_loss * f, global_step)
        writer.add_scalar('06 car up loss', self.rc_up_loss * f, global_step)
        writer.add_scalar('07 car velocity loss', self.rc_vel_loss * f, global_step)
        writer.add_scalar('08 car angular velocity loss', self.rc_ang_vel_loss * f, global_step)
        writer.add_scalar('09 car on ground loss', self.rc_on_ground_loss * f, global_step)
        writer.add_scalar('10 car ball touch loss', self.rc_ball_touch_loss * f, global_step)
        writer.add_scalar('11 car has jump loss', self.rc_has_jump_loss * f, global_step)
        writer.add_scalar('12 car has flip loss', self.rc_has_flip_loss * f, global_step)
        writer.add_scalar('13 car is demo loss', self.rc_is_demo_loss * f, global_step)
        writer.add_scalar('01 ball position diff', self.rb_pos_diff * f, global_step)
        writer.add_scalar('02 ball velocity diff', self.rb_vel_diff * f, global_step)
        writer.add_scalar('03 ball angular velocity diff', self.rb_ang_vel_diff * f, global_step)
        writer.add_scalar('04 car position diff', self.rc_pos_diff * f, global_step)
        writer.add_scalar('05 car forward similarity', self.rc_forward_sim * f, global_step)
        writer.add_scalar('06 car up similarity', self.rc_up_sim * f, global_step)
        writer.add_scalar('07 car velocity diff', self.rc_vel_diff * f, global_step)
        writer.add_scalar('08 car angular velocity diff', self.rc_ang_vel_diff * f, global_step)
        writer.add_scalar('09 car on ground accuracy', self.rc_on_ground_acc * f, global_step)
        writer.add_scalar('10 car ball touch accuracy', self.rc_ball_touch_acc * f, global_step)
        writer.add_scalar('11 car has jump accuracy', self.rc_has_jump_acc * f, global_step)
        writer.add_scalar('12 car has flip accuracy', self.rc_has_flip_acc * f, global_step)
        writer.add_scalar('13 car is demo accuracy', self.rc_is_demo_acc * f, global_step)

        self.reset_running_losses()
