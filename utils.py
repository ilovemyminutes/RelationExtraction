import pickle
import datetime
import random
import numpy as np
import torch


def load_pickle(path: str):
    with open(path, "rb") as pkl_file:
        output = pickle.load(pkl_file)
    return output


def set_seed(seed: int = 42, contain_cuda: bool = False):
    random.seed(seed)
    np.random.seed(seed)

    if contain_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed set as {seed}")


def get_timestamp():
    KST = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(tz=KST)
    now2str = now.strftime("%Y%m%d%H%M%S")
    return now2str

def verbose(phase: str, eval: dict, loss: float):
    acc = eval["accuracy"]
    f1 = eval["f1"]
    prc = eval["precision"]
    rec = eval["recall"]
    print(
        f"[{phase}] ACC: {acc:.4f} F1: {f1:.4f} PRC: {prc:.4f} REC: {rec:.4f} Loss: {loss:.4f}"
    )

def ckpt_name(model_type, pretrained_type, epoch, score, loss, timestamp):
    name = f"{model_type}_{pretrained_type}_ep({epoch:0>2d})acc({score:.4f})loss({loss:.4f})id({timestamp}).pth"
    return name