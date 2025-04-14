# utils/logger.py
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir="runs/exp"):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
