import numpy as np
import torch


def mirror_state(obs):
    """
    Spiegelung an der Längslinie von Tor zu Tor
    :param obs:
    :return:
    """
    mirrored_obs = torch.zeros(obs.shape, device=obs.device)
    # ball
    coefs = torch.tensor([
        # ball
        -1, 1, 1,  # pos
        -1, 1, 1,  # vel
        1, -1, -1,  # ang_vel
        # car1
        -1, 1, 1,  # pos
        1, -1, -1,  # forward
        1, -1, -1,  # up
        -1, 1, 1,  # vel
        1, -1, -1,  # ang_vel
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # bools and timings
        1, -1, -1,  # forward flip
        1, -1, -1,  # up flip
        -1, 1, 1,  # vel flip
        1,  # pitch flip
        -1,  # yaw + roll flip
        # car2
        -1, 1, 1,  # pos
        1, -1, -1,  # forward
        1, -1, -1,  # up
        -1, 1, 1,  # vel
        1, -1, -1,  # ang_vel
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # bools and timings
        1, -1, -1,  # forward flip
        1, -1, -1,  # up flip
        -1, 1, 1,  # vel flip
        1,  # pitch flip
        -1,  # yaw + roll flip
        # inputs
        1, -1, 1, -1, -1, 1, 1, 1,  # car1
        1, -1, 1, -1, -1, 1, 1, 1  # car2
    ], device=obs.device)
    mirrored_obs[:85] = obs[:85] * coefs[:85]
    mirrored_obs[85:] = obs[85:] * coefs[85:obs.shape[0]]
    return mirrored_obs


def invert_state(obs):
    """
    Punktspiegelung (vertauschen von teams)
    :param obs:
    :return:
    """
    inverted_obs = torch.zeros(obs.shape, device=obs.device)
    # ball
    coefs = torch.tensor([
        # ball
        -1, -1, 1,  # pos
        -1, -1, 1,  # vel
        -1, 1, -1,  # ang_vel
        # car1
        -1, -1, 1,  # pos
        -1, 1, -1,  # forward
        -1, 1, -1,  # up
        -1, -1, 1,  # vel
        -1, 1, -1,  # ang_vel
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # bools and timings
        -1, 1, -1,  # forward
        -1, 1, -1,  # up
        -1, -1, 1,  # vel
        1,  # pitch flip
        1,  # yaw + roll flip
        # car2
        -1, -1, 1,  # pos
        -1, 1, -1,  # forward
        -1, 1, -1,  # up
        -1, -1, 1,  # vel
        -1, 1, -1,  # ang_vel
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # bools and timings
        -1, 1, -1,  # forward
        -1, 1, -1,  # up
        -1, -1, 1,  # vel
        1,  # pitch flip
        1,  # yaw + roll flip
        # inputs
        1, 1, 1, 1, 1, 1, 1, 1,  # car1
        1, 1, 1, 1, 1, 1, 1, 1  # car2
    ], device=obs.device)
    inverted_obs[:85] = obs[:85] * coefs[:85]
    inverted_obs[85:] = obs[85:] * coefs[85:obs.shape[0]]
    return inverted_obs


def invert_and_mirror_state(obs):
    """
    Spiegelung an der Mittellinie
    :param obs:
    :return:
    """
    return invert_state(mirror_state(obs))
