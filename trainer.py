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

        self.fc1 = nn.Linear(config.in_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, config.out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


def start_training(config):
    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)
    writer = SummaryWriter(config.tensorboard_path)
    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
    log(f'Device: {device}')

    # 0) prepare data
    log("Training Data")
    train_dataset = KickoffEnsemble(config.train_path, config)
    n_training_samples = len(train_dataset)
    """
    log("Test Data")
    test_dataset = KickoffEnsemble(config.test_path, config)
    n_test_samples = len(test_dataset)
    test_loader = DataLoader(dataset=test_dataset, batch_size=n_test_samples, num_workers=0, pin_memory=True)
    """


    n_iterations = math.ceil(n_training_samples / config.batch_size)

    # 1) setup model and optimizer
    start_epoch = 0
    model = Model(config)
    log(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if not os.path.exists(config.checkpoint_path):
        os.mkdir(config.checkpoint_path)
    else:
        checkpoints = os.listdir(config.checkpoint_path)
        if len(checkpoints) > 0:
            log("Found existing model with that name, continue from latest checkpoint")
            loaded_checkpoint = torch.load(os.path.join(config.checkpoint_path, checkpoints[len(checkpoints)-1]), map_location="cuda:0")
            start_epoch = loaded_checkpoint["epoch"]
            device = loaded_checkpoint["device"]
            model.load_state_dict(loaded_checkpoint["model_state"])
            optimizer.load_state_dict(loaded_checkpoint["optim_state"])
    model.to(device)

    free_mem, total_mem = torch.cuda.mem_get_info()
    batch_size = free_mem / (32*config.in_size)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              generator=torch.Generator(device=device))
    log(f"Total Memory: {total_mem}, Free Memory: {free_mem} Batch Size: {batch_size}")

    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print(f'Cuda Memory: total: {t} reserved: {r} allocated: {a} free: {f}')
    log(torch.cuda.mem_get_info())

    # 2) loss
    criterion = nn.MSELoss()

    # 3) training loop
    epochs_since_last_checkpoint = 0
    iteration = 0
    steps = 0
    steps_last_log = 0
    iterations_last_log = 0
    for epoch in range(start_epoch, config.num_epoch):
        running_loss = 0.0
        # log(f'Epoch {epoch}')
        for i, (x_train, y_train) in enumerate(train_loader):
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r - a  # free inside reserved
            print(f'Cuda Memory: total: {t} reserved: {r} allocated: {a} free: {f}')
            log(torch.cuda.mem_get_info())
            # forward pass and loss

            y_predicted = model(x_train)
            loss = criterion(y_predicted, y_train)
            running_loss += loss.item()

            # zero gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # updates
            optimizer.step()

            steps += x_train.shape[0]
            steps_last_log += x_train.shape[0]
            iterations_last_log += 1

            if steps_last_log >= config.steps_per_log:
                log(f'epoch: {epoch + 1}, step: {steps}, loss = {running_loss / iterations_last_log:.10f}')
                writer.add_scalar('training loss', running_loss / iterations_last_log, steps)
                running_loss = 0.0
                steps_last_log = 0
                iterations_last_log = 0
                """with torch.no_grad():
                    tensor_log(config, writer, y_predicted, y_train, steps)"""

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
            file_name = "cp_" + str(steps) + ".pth"
            torch.save(checkpoint, os.path.join(config.checkpoint_path, file_name))
            log(f'iteration: {iteration}, saved checkpoint as {file_name}')


def tensor_log(config, writer, y_pred, y_true, global_step):
    M = Metrics(config)
    if config.ball_out:
        writer.add_scalar('ball position loss', M.ball_pos_loss(y_pred, y_true), global_step)
        writer.add_scalar('ball velocity loss', M.ball_lin_vel_loss(y_pred, y_true), global_step)
        writer.add_scalar('ball angular velocity loss', M.ball_ang_vel_loss(y_pred, y_true), global_step)
    if config.num_car_out > 0:
        writer.add_scalar('car position loss', M.car_pos_loss(y_pred, y_true), global_step)
        writer.add_scalar('car forward rotation loss', M.car_forward_loss(y_pred, y_true), global_step)
        writer.add_scalar('car up rotation loss', M.car_up_loss(y_pred, y_true), global_step)
        writer.add_scalar('car velocity loss', M.car_lin_vel_loss(y_pred, y_true), global_step)
        writer.add_scalar('car angular velocity loss', M.car_ang_vel_loss(y_pred, y_true), global_step)
        writer.add_scalar('car on ground accuracy', M.car_on_ground_acc(y_pred, y_true), global_step)
        writer.add_scalar('car ball touch accuracy', M.car_ball_touch_acc(y_pred, y_true), global_step)
        writer.add_scalar('car has jump accuracy', M.car_has_jump_acc(y_pred, y_true), global_step)
        writer.add_scalar('car has flip accuracy', M.car_has_flip_acc(y_pred, y_true), global_step)
        writer.add_scalar('car is demo accuracy', M.car_is_demo_acc(y_pred, y_true), global_step)
