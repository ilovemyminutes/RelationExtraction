from torch import nn, optim
from adamp import AdamP
from transformers import AdamW, get_linear_schedule_with_warmup
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
    elif type == Optimizer.AdamW:
        optimizer = AdamW(model.parameters(), lr=lr)
    return optimizer


def get_scheduler(type: str, optimizer, num_training_steps: int):
    if type == Optimizer.CosineAnnealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    elif type == Optimizer.LambdaLR:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
    else:
        raise NotImplementedError()

    return scheduler
