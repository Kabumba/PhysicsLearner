import torch
import torch.nn as nn
import torch.nn.functional as F

from RocketleagueModel import RocketLeagueModel


class IndependentSplit(RocketLeagueModel):
    def __init__(self, config):
        super(IndependentSplit, self).__init__(config)
        car_size1 = 200
        car_size2 = 60

        self.fc_bpos1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_bpos2 = nn.Linear(self.config.hidden_size, 3)
        self.fc_bvel1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_bvel2 = nn.Linear(self.config.hidden_size, 3)
        self.fc_bangvel1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_bangvel2 = nn.Linear(self.config.hidden_size, 3)
        self.fc_cpos1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_cpos2 = nn.Linear(self.config.hidden_size, 3)
        self.fc_cvel1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_cvel2 = nn.Linear(self.config.hidden_size, 3)
        self.fc_cangvel1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_cangvel2 = nn.Linear(self.config.hidden_size, 3)

        self.fc_shared = nn.Linear(config.in_size, car_size1)
        self.fc_car1 = nn.Linear(car_size1, car_size2)
        self.fc_car_rot = nn.Linear(car_size2, 6)  # forward, up
        self.fc_car_bool = nn.Linear(car_size2, 5)

    def forward(self, x):
        bpos = F.relu(self.fc_bpos1(x))
        bvel = F.relu(self.fc_bvel1(x))
        bangvel = F.relu(self.fc_bangvel1(x))
        cpos = F.relu(self.fc_cpos1(x))
        cvel = F.relu(self.fc_cvel1(x))
        cangvel = F.relu(self.fc_cangvel1(x))

        bpos = self.fc_bpos2(bpos)
        bvel = self.fc_bvel2(bvel)
        bangvel = self.fc_bangvel2(bangvel)
        cpos = self.fc_cpos2(cpos)
        cvel = self.fc_cvel2(cvel)
        cangvel = self.fc_cangvel2(cangvel)

        x = F.relu(self.fc_shared(x))
        xc = F.relu(self.fc_car1(x))
        xc_rot = self.fc_car_rot(xc)
        xc_bool = torch.sigmoid((self.fc_car_bool(xc)))
        return bpos, bvel, bangvel, cpos, xc_rot, cvel, cangvel, xc_bool

    def format_prediction(self, y_predicted):
        bpos, bvel, bangvel, cpos, xc_rot, cvel, cangvel, xc_bool = y_predicted
        return bpos, bvel, bangvel, cpos, xc_rot[:, 0:3], xc_rot[:, 3:6], cvel, cangvel, xc_bool[:, 0], xc_bool[:, 1], xc_bool[:, 2], xc_bool[:, 3], xc_bool[:, 4]

    def accumulate_loss(self, b_pos_loss, b_vel_loss, b_ang_vel_loss, c_pos_loss, c_forward_loss, c_up_loss, c_vel_loss,
                        c_ang_vel_loss, c_on_ground_loss, c_ball_touch_loss, c_has_jump_loss, c_has_flip_loss, c_is_demo_loss):
        mse = b_pos_loss + b_vel_loss + b_ang_vel_loss + c_pos_loss + c_vel_loss + c_ang_vel_loss + (c_up_loss + c_forward_loss) / 2
        bce = c_on_ground_loss + c_ball_touch_loss + c_has_jump_loss + c_has_flip_loss + c_is_demo_loss
        loss = mse + bce
        return loss