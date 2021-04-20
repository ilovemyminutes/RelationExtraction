import os
import pickle
import json
import datetime
import random
import numpy as np
import torch


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def load_pickle(path: str):
    with open(path, "rb") as pkl_file:
        output = pickle.load(pkl_file)
    return output


def save_pickle(path: str, f: object) -> None:
    with open(path, "wb") as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(path: str, f: object) -> None:
    with open(path, "w") as json_path:
        json.dump(
            f,
            json_path,
        )


def load_json(path: str) -> dict:
    with open(path, "r") as json_file:
        output = json.load(json_file)
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
    name = f"{model_type}_{os.path.basename(pretrained_type)}_ep({epoch:0>2d})acc({score:.4f})loss({loss:.4f})id({timestamp}).pth"
    return name
