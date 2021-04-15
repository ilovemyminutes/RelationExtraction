import os
from dataclasses import dataclass

from tokenizers.implementations.base_tokenizer import BaseTokenizer


TRAIN = "./input/data/train/train.tsv"
TEST = "./input/data/test/test.tsv"
LABEL = "./input/data/label_type.pkl"
SAVEPATH = "./saved_models"

DOT = "."


@dataclass
class Config:
    """notebooks 디렉토리의 주피터 환경에서 아래의 configuration을 활용할 수 있도록 구성
    DOT + SOMETHING: '../something/something' <- 디렉토리 경로를 바꿔주게 됨
    """

    Train: str = TRAIN if os.path.isfile(TRAIN) else DOT + TRAIN
    Test: str = TEST if os.path.isfile(TEST) else DOT + TEST
    Label: str = LABEL if os.path.isfile(LABEL) else DOT + LABEL
    SavePath: str = SAVEPATH if os.path.isfile(SAVEPATH) else DOT + SAVEPATH
    NumClasses: int = 42


@dataclass
class TokenizationType:
    Base: str = "Base"
    EM: str = "EM"  # Entity Marking


@dataclass
class ModelType:
    BertMultiLingual: str = "bert-base-multilingual-cased"


class TrainArgs:
    Base: dict = dict(
        output_dir=Config.SavePath,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        seed=42,
        num_labels=Config.NumClasses,
    )

# def train_args(type: str=TrainArgs.Base):
#     if type == TrainArgs.Base:
#         args = TrainArgs.Base
#     else:
#         raise NotImplementedError
#     return args
