import os
from dataclasses import dataclass
import torch


TRAIN = "./input/data/train/train.tsv"
TEST = "./input/data/test/test.tsv"
LABEL = "./input/data/label_type.pkl"
SAVEPATH = "./saved_models"
LOGS = "./logs"
CKPT = "./saved_models"

DOT = "."


@dataclass
class Config:
    """notebooks 디렉토리의 주피터 환경에서 아래의 configuration을 활용할 수 있도록 구성
    DOT + SOMETHING: '../something/something' <- 디렉토리 경로를 바꿔주게 됨
    """

    Train: str = TRAIN if os.path.isfile(TRAIN) else DOT + TRAIN
    Test: str = TEST if os.path.isfile(TEST) else DOT + TEST
    ValidSize: float = 0.2
    Label: str = LABEL if os.path.isfile(LABEL) else DOT + LABEL
    SavePath: str = SAVEPATH if os.path.isfile(SAVEPATH) else DOT + SAVEPATH
    Logs: str = LOGS if os.path.isfile(LOGS) else DOT + LOGS
    NumClasses: int = 42
    Epochs: int = 1
    Batch32: int = 32
    Batch64: int = 64
    LR: float = 0.001
    Seed: int = 42
    Device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CheckPoint: str = CKPT if os.path.isfile(CKPT) else DOT + CKPT


@dataclass
class Optimizer:
    Adam: str = "Adam"
    AdamP: str = "AdamP"
    SGD: str = "SGD"
    Momentum: str = "Momentum"
    CosineScheduler: str = "CosineScheduler"


@dataclass
class Loss:
    CE: str = "crossentropyloss"
    LS: str = "labelsmoothingLoss"


@dataclass
class TokenizationType:
    Base: str = "Base"
    EM: str = "EM"  # Entity Marking


@dataclass
class ModelType:
    Base: str = "BertModel"
    SequenceClf: str = "BertForSequenceClassification"


@dataclass
class PreTrainedType:
    BertMultiLingual: str = "bert-base-multilingual-cased"


class TrainArgs:
    Base: dict = dict(
        output_dir=Config.SavePath,
        overwrite_output_dir=True,
        num_train_epochs=Config.Epochs,
        per_device_train_batch_size=Config.Batch32,
        per_device_eval_batch_size=Config.Batch64,
        warmup_steps=500,
        weight_decay=0.01,
        seed=Config.Seed,
        logging_dir=Config.Logs,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
    )
