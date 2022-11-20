import torch
import torch.nn as nn
import torch.nn.functional as F

from RocketleagueModel import RocketLeagueModel


class Split(RocketLeagueModel):
    def __init__(self, config):
        super(Split, self).__init__(config)
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

    def forward(self, x):
        x = F.relu(self.fc_shared(x))
        xb = F.relu(self.fc_ball1(x))
        xb = self.fc_ball2(xb)
        xc = F.relu(self.fc_car1(x))
        xc1 = self.fc_car2(xc)
        xc_rot = torch.sin(self.fc_car_rot(xc))
        xc_bool = torch.sigmoid((self.fc_car_bool(xc)))
        return xb, xc1, xc_rot, xc_bool

    def format_prediction(self, y_predicted):
        a, b, c, d = y_predicted
        b1 = a[:, 0:3]
        b2 = a[:, 3:6]
        b3 = a[:, 6:9]
        c1 = b[:, 0:3]
        c2 = c[:, 0:3]
        c3 = c[:, 3:6]
        c4 = b[:, 3:6]
        c5 = b[:, 6:9]
        c6 = d[:, 0]
        c7 = d[:, 1]
        c8 = d[:, 2]
        c9 = d[:, 3]
        c10 = d[:, 4]
        return b1, b2, b3, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10

    def accumulate_loss(self, b_pos_loss, b_vel_loss, b_ang_vel_loss, c_pos_loss, c_forward_loss, c_up_loss, c_vel_loss,
                        c_ang_vel_loss,
                        c_on_ground_loss, c_ball_touch_loss, c_has_jump_loss, c_has_flip_loss, c_is_demo_loss):
        mse = (b_pos_loss + b_vel_loss + b_ang_vel_loss) / 3 + (c_pos_loss + c_vel_loss + c_ang_vel_loss) / 3 + (
                    c_up_loss + c_forward_loss) / 2
        bce = c_on_ground_loss + c_ball_touch_loss + c_has_jump_loss + c_has_flip_loss + c_is_demo_loss
        loss = mse + bce
        return loss
