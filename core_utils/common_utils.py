import numpy as np
import torch
import random
import os
import collections


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_cuda(*tensors, device="cuda"):
    if device == "cpu" or not torch.cuda.is_available():
        return list(tensors)
    else:
        return [tensor.to(device) for tensor in tensors]


def optimizer_zero_grad(*optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()


def optimizer_step(*optimizers, scaler=None):
    for optimizer in optimizers:
        if scaler is not None:
            scaler.step(optimizer)
        else:
            optimizer.step()


def lr_scheduler_step(*lr_scheduler):
    for scheduler in lr_scheduler:
        scheduler.step()


def result_append(results, metric_name, value):
    if metric_name not in results:
        results[metric_name] = []
    results[metric_name].append(value)


def makedir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    return False
