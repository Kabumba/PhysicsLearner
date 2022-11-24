import os.path

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
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)
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
    train_dataset = KickoffEnsemble(config.train_path, config)

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
    model.load_checkpoint()
    log(model)

    # setup DataLoader
    free_mem, total_mem = torch.cuda.mem_get_info()
    # mem_buffer = 1050000000
    # batch_size = math.floor((free_mem - mem_buffer) / (32 * config.in_size))
    batch_size = config.batch_size
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=config.num_workers,
                              pin_memory=config.pin_memory,
                              # generator=torch.Generator(device=device)
                              )
    log(f"Total Memory: {total_mem}B, Free Memory: {free_mem}B, Batch Size: {batch_size}, Worker: {config.num_workers}")

    # 3) training loop
    steps_last_log = 0
    steps_last_tensor_log = 0
    steps_last_checkpoint = 0
    iterations_last_log = 0
    iterations_last_tensor_log = 0
    running_loss = 0.0
    max_epoch = 100000

    log(f"-------------- Start Training! --------------")
    for epoch in range(start_epoch, max_epoch):
        for i, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            # forward pass and loss

            y_predicted = model(x_train)
            loss = model.criterion(y_predicted, y_train, x_train)
            running_loss += loss.item()

            # zero gradients
            model.zero_grad()

            # backward pass
            loss.backward()

            # updates
            model.optim_step()
            model.update_training_steps(x_train.shape[0])

            del y_train
            del y_predicted

            steps += x_train.shape[0]
            steps_last_log += x_train.shape[0]
            steps_last_tensor_log += x_train.shape[0]
            steps_last_checkpoint += x_train.shape[0]
            iterations_last_log += 1
            iterations_last_tensor_log += 1

            del x_train

            # tensorboard logs
            if steps_last_tensor_log >= config.steps_per_tensor_log:
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
        if steps >= config.max_steps:
            break
    log(f"Model has trained for {config.max_steps} steps. Run over.")
