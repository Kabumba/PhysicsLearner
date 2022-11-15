import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from kickoff_dataset import KickoffEnsemble
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
    model = Model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loaded_checkpoint = False

    # load checkpoint
    if not os.path.exists(config.checkpoint_path):
        os.mkdir(config.checkpoint_path)
    else:
        checkpoints = os.listdir(config.checkpoint_path)
        paths = [os.path.join(config.checkpoint_path, basename) for basename in checkpoints]
        if config.continue_from_checkpoint and len(checkpoints) > 0:
            cp_file = max(paths, key=os.path.getctime)
            log(f"Found existing model with that name, continue from latest checkpoint {cp_file}")
            loaded_checkpoint = torch.load(cp_file, map_location="cuda:0")
            start_epoch = loaded_checkpoint["epoch"]
            steps = loaded_checkpoint["step"]
            device = loaded_checkpoint["device"]
            model.load_state_dict(loaded_checkpoint["model_state"])
            optimizer.load_state_dict(loaded_checkpoint["optim_state"])
            loaded_checkpoint = True
    model.to(device)
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

    # 2) loss
    criterion = nn.MSELoss()

    # 3) training loop
    steps_last_log = 0
    steps_last_tensor_log = 0
    steps_last_checkpoint = 0
    last_loss = 0
    iterations_last_log = 0
    iterations_last_tensor_log = 0
    running_loss = 0.0
    running_tensor_loss = 0.0
    first_log = True
    max_epoch = 100000

    log(f"-------------- Start Training! --------------")
    if not loaded_checkpoint:
        writer.add_scalar('training loss', 1, 0)
    for epoch in range(start_epoch, max_epoch):
        for i, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            # forward pass and loss

            y_predicted = model(x_train)
            loss = criterion(y_predicted, y_train)
            running_loss += loss.item()
            running_tensor_loss += loss.item()

            # zero gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # updates
            optimizer.step()
            del y_train

            steps += x_train.shape[0]
            steps_last_log += x_train.shape[0]
            steps_last_checkpoint += x_train.shape[0]
            iterations_last_log += 1
            iterations_last_tensor_log += 1

            del x_train

            # tensorboard logs
            if steps_last_tensor_log >= config.steps_per_tensor_log:
                logged_loss = running_tensor_loss / iterations_last_tensor_log
                writer.add_scalar('training loss', logged_loss, steps)
                if not first_log:
                    delta_loss = logged_loss - last_loss
                    writer.add_scalar('delta loss', delta_loss, steps)
                else:
                    first_log = False
                last_loss = logged_loss
                running_tensor_loss = 0.0
                steps_last_tensor_log = 0
                iterations_last_tensor_log = 0
                """with torch.no_grad():
                    tensor_log(config, writer, y_predicted, y_train, steps)"""

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
                checkpoint = {
                    "epoch": epoch,
                    "device": device,
                    "step": steps,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict()
                }
                file_name = "cp_" + str(steps) + ".pth"
                torch.save(checkpoint, os.path.join(config.checkpoint_path, file_name))
                log(f'step: {steps}, saved checkpoint as {file_name}')

            if steps >= config.max_steps:
                break
        if steps >= config.max_steps:
            break
    log(f"Model has trained for {config.max_steps} steps. Run over.")


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
