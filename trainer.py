import math
import os.path
import random
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from IndependentSplit import IndependentSplit
from kickoff_dataset import KickoffEnsemble
from logger import log

from independent3 import Independent3
from independent1 import Independent
from independent2 import Independent2
from naive import Naive
from partition import create_partition
from priority_sampler import PrioritySampler
from split import Split

"""
1) Design model (input, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: compute prediction and loss
    - backward pass: gradients
    - update weights
"""


# 1) model


def start_training(config):
    # setup directories
    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)
    if not os.path.exists(config.log_path):
        os.mkdir(config.log_path)
    parts = os.listdir(config.tensorboard_path)
    current_part = len(parts)
    tensorboard_path = os.path.join(config.tensorboard_path, "part" + str(current_part))
    writer = SummaryWriter(tensorboard_path)

    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")"""
    log(f'Device: {device}')

    # 0) prepare data
    log("Training Data")

    # 1) setup model and optimizer
    start_epoch = 0
    steps = 0
    model = None
    if config.model_type == "Naive":
        model = Naive(config)
    if config.model_type == "Split":
        model = Split(config)
    if config.model_type == "Independent":
        model = Independent(config)
    if config.model_type == "Independent2":
        model = Independent2(config)
    if config.model_type == "Independent3":
        model = Independent3(config)
    if config.model_type == "IndependentSplit":
        model = IndependentSplit(config)
    if config.model_type is None:
        raise ValueError(f'{config.model_type} is not a supported model type!')
    model.to(device)
    model.init_optim()

    # load checkpoint
    start_epoch, steps = model.load_checkpoint()
    log(model)

    # setup DataLoader
    free_mem, total_mem = torch.cuda.mem_get_info()
    batch_size = config.batch_size
    if config.num_partitions == 1:
        train_dataset = KickoffEnsemble(config.train_path, None, config)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size - config.loss_feedback,
                                  # batch_sampler=PrioritySampler(train_dataset, batch_size, 2),
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=config.pin_memory,
                                  # generator=torch.Generator(device=device)
                                  )

    log(f"Total Memory: {total_mem}B, Free Memory: {free_mem}B, Batch Size: {batch_size}, Loss Feedback: {config.loss_feedback}, Worker: {config.num_workers}")

    # 3) training loop
    steps_last_log = 0
    steps_last_tensor_log = 0
    steps_last_checkpoint = 0
    iterations_last_log = 0
    iterations_last_tensor_log = 0
    running_loss = 0.0
    max_epoch = 100000
    num_partitions = config.num_partitions
    feedback_batch = torch.tensor([], device=device), torch.tensor([], device=device)

    log(f"-------------- Start Training! --------------")
    for epoch in range(start_epoch, max_epoch):
        partitions = [None]
        if num_partitions > 1:
            partitions = create_partition(config, num_partitions)
        for partition in partitions:
            if num_partitions > 1:
                train_dataset = KickoffEnsemble(config.train_path, partition, config)
                train_loader = DataLoader(dataset=train_dataset,
                                          batch_size=batch_size - config.loss_feedback,
                                          # batch_sampler=PrioritySampler(train_dataset, batch_size, 2),
                                          shuffle=True,
                                          num_workers=config.num_workers,
                                          pin_memory=config.pin_memory,
                                          # generator=torch.Generator(device=device)
                                          )
            for i, ((x_train, y_train), indices) in enumerate(train_loader):
                x_train, y_train = x_train.to(device, non_blocking=True), y_train.to(device, non_blocking=True)
                if config.loss_feedback > 0:
                    x_feedback, y_feedback = feedback_batch
                    x_train = torch.cat((x_train, x_feedback))
                    y_train = torch.cat((y_train, y_feedback))
                # forward pass and loss
                for j in range(config.batch_repetitions):
                    y_predicted = model(x_train)
                    loss, feedback_batch = model.criterion(y_predicted, y_train, x_train)
                    del y_predicted
                    running_loss += loss.item()

                    # zero gradients
                    model.optim_zero_grad()

                    # backward pass
                    loss.backward()

                    # updates
                    model.optim_step()
                    model.update_training_steps(x_train.shape[0])

                    steps += x_train.shape[0]
                    steps_last_log += x_train.shape[0]
                    steps_last_tensor_log += x_train.shape[0]
                    steps_last_checkpoint += x_train.shape[0]
                    iterations_last_log += 1
                    iterations_last_tensor_log += 1

                    # tensorboard logs
                    if steps_last_tensor_log >= config.steps_per_tensor_log:
                        if not os.path.exists(tensorboard_path):
                            os.mkdir(tensorboard_path)
                        model.tensor_log(writer, iterations_last_tensor_log, steps)
                        steps_last_tensor_log = 0
                        iterations_last_tensor_log = 0

                    # printed logs
                    if steps_last_log >= config.steps_per_log:
                        logged_loss = running_loss / iterations_last_log
                        log(f'epoch: {epoch + 1}, step: {steps}, loss = {logged_loss:.10f}')
                        running_loss = 0.0
                        steps_last_log = 0
                        iterations_last_log = 0

                    # checkpoints
                    if steps >= config.max_steps or steps_last_checkpoint >= config.steps_per_checkpoint:
                        steps_last_checkpoint = 0
                        model.save_checkpoint(epoch, device, steps)

                    if steps >= config.max_steps:
                        break
                del y_train
                del x_train
                if steps >= config.max_steps:
                    break
            if num_partitions > 1:
                del train_loader
                del train_dataset
            if steps >= config.max_steps:
                break
        if steps >= config.max_steps:
            break
    log(f"Model has trained for {config.max_steps} steps. Run over.")
