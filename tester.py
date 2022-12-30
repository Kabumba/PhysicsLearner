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
from auswertung import Auswerter

"""
1) Design model (input, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: compute prediction and loss
    - backward pass: gradients
    - update weights
"""


# 1) model
def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

def remove_suffix(string, suffix):
    if string.endswith(suffix):
        return string[:-len(suffix)]
    return string


def start_testing(configs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = KickoffEnsemble(configs[0].test_path, None, configs[0])
    log(f'Device: {device}')
    first = True
    for config in configs:
        start_time = time.time()
        log(f"{config.name}")
        if first:
            first = False
        else:
            test_dataset.set_new_config(config)
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
        log("Test Data")

        # 1) setup model and optimizer
        models = []
        cp_dir = config.checkpoint_path
        files = os.listdir(cp_dir)
        cphs = []
        dirs = []
        for f in files:
            if f.endswith(".cph"):
                cphs.append(f)
            else:
                dirs.append(f)
        cp_indices = []
        for f in cphs:
            cp = remove_prefix(f, "cph_")
            cp = remove_suffix(cp, ".cph")
            cp_indices.append(int(cp))
        cp_indices.sort()
        for i in range(len(cp_indices)):
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
            start_epoch, steps = model.load_checkpoint(i)
            models.append(model)

        # setup DataLoader
        free_mem, total_mem = torch.cuda.mem_get_info()
        batch_size = config.batch_size
        pin_memory = config.pin_memory # and not lido
        num_workers = config.num_workers # if not lido else 0
        n = len(test_dataset)
        test_loader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  # batch_sampler=PrioritySampler(train_dataset, batch_size, 2),
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  # generator=torch.Generator(device=device)
                                  )

        log(f"Total Memory: {total_mem}B, Free Memory: {free_mem}B, Batch Size: {batch_size}, Loss Feedback: {config.loss_feedback}, Worker: {config.num_workers}")

        iterations_last_tensor_log = 0
        auswerter = Auswerter(device)
        progress = 0
        percent = 0


        log(f"-------------- Start Testing! --------------")
        for _, ((x_test, y_test), indices) in enumerate(test_loader):
            x_test, y_test = x_test.to(device, non_blocking=True), y_test.to(device, non_blocking=True)
            for i in range(len(cp_indices)):
                model = models[i]
                y_predicted = model(x_test)
                _, _ = model.criterion(y_predicted, y_test, x_test)
                iterations_last_tensor_log += 1
                if i == len(cp_indices)-1:
                    #Ausführliche Auswertung des letzten checkpoints
                    auswerter.gather_info(y_predicted, y_test, x_test)
            progress += x_test.shape[0]
            del y_predicted
            del y_test
            del x_test
            if percent - 100*progress/n > 0.01:
                percent = 100*progress/n
                log(f"Verarbeitet: {percent:.2f}%")
        auswerter.save()
        for i in range(len(cp_indices)):
            # tensorboard logs
            model = models[i]
            steps = cp_indices[i]
            if not os.path.exists(tensorboard_path):
                os.mkdir(tensorboard_path)
            model.tensor_log(writer, iterations_last_tensor_log, steps)
    log("done")