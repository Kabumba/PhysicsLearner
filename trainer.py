import os.path
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Independent4 import Independent4
from IndependentSplit import IndependentSplit
from independentScalar import IndependentScalar
from kickoff_dataset import KickoffEnsemble
from logger import log

from independent3 import Independent3
from independent1 import Independent
from independent2 import Independent2
from naive import Naive
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


def start_training(configs, lido):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = KickoffEnsemble(configs[0].train_path, None, configs[0])
    log(f'Device: {device}')
    first = True
    for config in configs:
        start_time = time.time()
        log(f"{config.name}")
        if first:
            first = False
        else:
            train_dataset.set_new_config(config)
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
        if config.model_type == "Independent4":
            model = Independent4(config)
        if config.model_type == "IndependentSplit":
            model = IndependentSplit(config)
        if config.model_type == "IndependentScalar":
            model = IndependentScalar(config)
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
        pin_memory = config.pin_memory # and not lido
        num_workers = config.num_workers # if not lido else 0
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size - config.loss_feedback,
                                  # batch_sampler=PrioritySampler(train_dataset, batch_size, 2),
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
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
        feedback_batch = torch.tensor([], device=device), torch.tensor([], device=device)

        log(f"-------------- Start Training! --------------")
        for epoch in range(start_epoch, max_epoch):
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
                    if time.time() - start_time > config.max_minutes * 60:
                        break
                del y_train
                del x_train
                if steps >= config.max_steps:
                    break
                if time.time() - start_time > config.max_minutes * 60:
                    break
            if steps >= config.max_steps:
                break
            if time.time() - start_time > config.max_minutes * 60:
                break
        log(f"{config.name} has trained for {steps} steps. Run over.")
        del train_loader
