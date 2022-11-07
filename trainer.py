import math
import os.path

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from kickoff_dataset import KickoffDataset, KickoffEnsemble
from logger import log
from metrics import Metrics

"""
1) Design model (input, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: compute prediction and loss
    - backward pass: gradients
    - update weights
"""


# 1) model
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.fc = []
        if config.num_hidden_layers == 0:
            self.fc.append(nn.Linear(config.in_size, config.out_size))
        else:
            self.fc.append(nn.Linear(config.in_size, config.hidden_size))
            for i in range(config.num_hidden_layers - 1):
                self.fc.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.fc.append(nn.Linear(config.hidden_size, config.out_size))

    def forward(self, x):
        num_layers = len(self.fc) - 1
        for i in range(num_layers):
            x = F.relu(self.fc[i](x))
        x = self.fc[num_layers](x)
        return x


def start_training(config):
    writer = SummaryWriter(config.result_path)
    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 0) prepare data
    train_dataset = KickoffEnsemble(config.train_path)
    n_training_samples = len(train_dataset)
    test_dataset = KickoffEnsemble(config.test_path)
    n_test_samples = len(test_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=n_test_samples, num_workers=0, pin_memory=True)

    n_iterations = math.ceil(n_training_samples / config.batch_size)

    # 1) setup model and optimizer
    start_epoch = 0
    model = Model(config)
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if not os.path.exists(config.checkpoint_path):
        os.mkdir(config.checkpoint_path)
    else:
        checkpoints = os.listdir(config.checkpoint_path)
        log("Found existing model with that name, continue from latest checkpoint")
        loaded_checkpoint = torch.load(os.path.join(config.checkpoint_path, checkpoints[len(checkpoints)]))
        start_epoch = loaded_checkpoint["epoch"]
        device = loaded_checkpoint["device"]
        model.load_state_dict(loaded_checkpoint["model_state"])
        optimizer.load_state_dict(loaded_checkpoint["optim_state"])
    model.to(device)
    writer.add_graph(model)

    # 2) loss
    criterion = nn.MSELoss()

    # 3) training loop
    epochs_since_last_checkpoint = 0
    iteration = 0
    for epoch in range(start_epoch, config.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # forward pass and loss
            x_train, y_train = data
            y_predicted = model(x_train)
            loss = criterion(y_predicted, y_train)
            running_loss += loss.item()

            # backward pass
            loss.backward()

            # updates
            optimizer.step()

            # zero gradients
            optimizer.zero_grad()

            if iteration % config.iterations_per_log == 0:
                log(f'epoch: {epoch + 1}, iteration: {iteration}, loss = {loss.item():.10f}')
                writer.add_scalar('training loss', running_loss / config.batch_size, iteration * config.batch_size)
                running_loss = 0.0

            iteration += 1

        epochs_since_last_checkpoint += 1
        if epochs_since_last_checkpoint == config.epochs_per_checkpoint:
            epochs_since_last_checkpoint = 0
            checkpoint = {
                "epoch": epoch,
                "device": device,
                "iteration": iteration,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict()
            }
            file_name = "cp_" + str(iteration) + ".pth"
            torch.save(checkpoint, os.path.join(config.checkpoint_path, file_name))
            log(f'iteration: {iteration}, saved checkpoint as {file_name}')

        # 4) evaluation
        with torch.no_grad():
            M = Metrics(config)
            global_step = n_training_samples * (epoch + 1)
            for data in test_loader:
                x_test, y_test = data
                y_pred = model(x_test)
                if config.ball_out:
                    writer.add_scalar('ball position loss', M.ball_pos_loss(y_pred, y_test), global_step)
                    writer.add_scalar('ball velocity loss', M.ball_lin_vel_loss(y_pred, y_test), global_step)
                    writer.add_scalar('ball angular velocity loss', M.ball_ang_vel_loss(y_pred, y_test), global_step)
                if config.num_cars_out > 0:
                    writer.add_scalar('car position loss', M.car_pos_loss(y_pred, y_test), global_step)
                    writer.add_scalar('car forward rotation loss', M.car_forward_loss(y_pred, y_test), global_step)
                    writer.add_scalar('car up rotation loss', M.car_up_loss(y_pred, y_test), global_step)
                    writer.add_scalar('car velocity loss', M.car_lin_vel_loss(y_pred, y_test), global_step)
                    writer.add_scalar('car angular velocity loss', M.car_ang_vel_loss(y_pred, y_test), global_step)
                    writer.add_scalar('car on ground accuracy', M.car_on_ground_acc(y_pred, y_test), global_step)
                    writer.add_scalar('car ball touch accuracy', M.car_ball_touch_acc(y_pred, y_test), global_step)
                    writer.add_scalar('car has jump accuracy', M.car_has_jump_acc(y_pred, y_test), global_step)
                    writer.add_scalar('car has flip accuracy', M.car_has_flip_acc(y_pred, y_test), global_step)
                    writer.add_scalar('car is demo accuracy', M.car_is_demo_acc(y_pred, y_test), global_step)
