import torch
import torch.nn as nn
import torch.nn.functional as F

from RocketleagueModel import RocketLeagueModel


class Independent3(RocketLeagueModel):
    def __init__(self, config):
        super(Independent3, self).__init__(config)

        self.fc_bpos1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_bpos2 = nn.Linear(self.config.hidden_size, 3)

        self.fc_bvel1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_bvel2 = nn.Linear(self.config.hidden_size, 3)

        self.fc_bangvel1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_bangvel2 = nn.Linear(self.config.hidden_size, 3)

        self.fc_cpos1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_cpos2 = nn.Linear(self.config.hidden_size, 3)

        self.fc_crot1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_cforward2 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_cforward3 = nn.Linear(self.config.hidden_size, 3)
        self.fc_cup2 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_cup3 = nn.Linear(self.config.hidden_size, 3)

        self.fc_cvel1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_cvel2 = nn.Linear(self.config.hidden_size, 3)

        self.fc_cangvel2 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_cangvel3 = nn.Linear(self.config.hidden_size, 3)

        self.fc_cbool1 = nn.Linear(config.in_size, self.config.hidden_size)
        self.fc_conground2 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_conground3 = nn.Linear(self.config.hidden_size, 1)

        self.fc_cballtouch2 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_cballtouch3 = nn.Linear(self.config.hidden_size, 1)

        self.fc_chasjump2 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_chasjump3 = nn.Linear(self.config.hidden_size, 1)

        self.fc_chasflip2 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_chasflip3 = nn.Linear(self.config.hidden_size, 1)

        self.fc_cisdemo2 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc_cisdemo3 = nn.Linear(self.config.hidden_size, 1)

    def forward(self, x):
        bpos = F.relu(self.fc_bpos1(x))
        bvel = F.relu(self.fc_bvel1(x))
        bangvel = F.relu(self.fc_bangvel1(x))
        cpos = F.relu(self.fc_cpos1(x))
        crot = F.relu(self.fc_crot1(x))
        cvel = F.relu(self.fc_cvel1(x))
        cbool = F.relu(self.fc_cbool1(x))

        cforward = F.relu(self.fc_cforward2(crot))
        cup = F.relu(self.fc_cup2(crot))
        cangvel = F.relu(self.fc_cangvel2(crot))
        conground = F.relu(self.fc_conground2(cbool))
        cballtouch = F.relu(self.fc_cballtouch2(cbool))
        chasjump = F.relu(self.fc_chasjump2(cbool))
        chasflip = F.relu(self.fc_chasflip2(cbool))
        cisdemo = F.relu(self.fc_cisdemo2(cbool))

        bpos = self.fc_bpos2(bpos)
        bvel = self.fc_bvel2(bvel)
        bangvel = self.fc_bangvel2(bangvel)
        cpos = self.fc_cpos2(cpos)
        cforward = torch.sin(self.fc_cforward3(cforward))
        cup = torch.sin(self.fc_cup3(cup))
        cvel = self.fc_cvel2(cvel)
        cangvel = self.fc_cangvel3(cangvel)
        conground = torch.sigmoid(self.fc_conground3(conground))
        cballtouch = torch.sigmoid(self.fc_cballtouch3(cballtouch))
        chasjump = torch.sigmoid(self.fc_chasjump3(chasjump))
        chasflip = torch.sigmoid(self.fc_chasflip3(chasflip))
        cisdemo = torch.sigmoid(self.fc_cisdemo3(cisdemo))

        return bpos, bvel, bangvel, cpos, cforward, cup, cvel, cangvel, conground, cballtouch, chasjump, chasflip, cisdemo

    def format_prediction(self, y_predicted):
        return y_predicted

    def accumulate_loss(self, b_pos_loss, b_vel_loss, b_ang_vel_loss, c_pos_loss, c_forward_loss, c_up_loss, c_vel_loss,
                        c_on_ground_loss, c_ball_touch_loss, c_has_jump_loss, c_has_flip_loss, c_is_demo_loss):
        mse = b_pos_loss + b_vel_loss + b_ang_vel_loss + c_pos_loss + c_forward_loss + c_up_loss + c_vel_loss
        bce = c_on_ground_loss + c_ball_touch_loss + c_has_jump_loss + c_has_flip_loss + c_is_demo_loss
        loss = mse + bce
        return loss