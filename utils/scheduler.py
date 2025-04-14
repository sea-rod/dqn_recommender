# utils/scheduler.py
from torch.optim.lr_scheduler import StepLR


def get_scheduler(optimizer, step_size=100, gamma=0.9):
    return StepLR(optimizer, step_size=step_size, gamma=gamma)
