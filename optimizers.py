from torch import nn, optim
from adamp import AdamP
from config import Optimizer


def get_optimizer(model: nn.Module, type: str, lr: float):
    if type == Optimizer.Adam:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif type == Optimizer.SGD:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif type == Optimizer.Momentum:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif type == Optimizer.AdamP:
        optimizer = AdamP(
            model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2
        )
    return optimizer


def get_scheduler(type: str, optimizer):
    if type == Optimizer.CosineScheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    else:
        raise NotImplementedError()

    return scheduler
