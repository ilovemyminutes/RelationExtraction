import os
from dataclasses import dataclass
import torch


TRAIN = "./input/data/train/train.tsv"
TRAIN_AUG = "./preprocessed/train_augmented.csv"
TRAIN_41 = "./preprocessed/train41.csv"
TRAIN_BIN = "./preprocessed/train_binary.csv"
TEST = "./input/data/test/test.tsv"
LABEL = "./input/data/label_type.pkl"
LABEL41 = "./input/data/label_type41.pkl"
LOGS = "./logs"
CKPT = "./saved_models"

DOT = "."


@dataclass
class Config:
    """notebooks 디렉토리의 주피터 환경에서 아래의 configuration을 활용할 수 있도록 구성
    DOT + SOMETHING: '../something/something' <- 디렉토리 경로를 바꿔주게 됨
    """

    Train: str = TRAIN if os.path.isfile(TRAIN) else DOT + TRAIN
    TrainP: str = TRAIN_AUG if os.path.isfile(TRAIN_AUG) else DOT + TRAIN_AUG
    Train41: str = TRAIN_41 if os.path.isfile(TRAIN_41) else DOT + TRAIN_41
    TrainBin: str = TRAIN_BIN if os.path.isfile(TRAIN_BIN) else DOT + TRAIN_BIN
    Test: str = TEST if os.path.isfile(TEST) else DOT + TEST
    SamplingWeights: str = './preprocessed/sampling_weights.pkl'
    SamplingWeightsP: str = './preprocessed/sampling_weights_modified.pkl'
    ValidSize: float = 0.1
    Label: str = LABEL if os.path.isfile(LABEL) else DOT + LABEL
    Label41: str = LABEL41 if os.path.isfile(LABEL41) else DOT + LABEL41
    Logs: str = LOGS if os.path.isfile(LOGS) else DOT + LOGS
    NumClasses: int = 42
    NumBinary: int = 2
    Num41: int = 41
    
    Epochs: int = 20

    Batch8: int = 8
    Batch16: int = 16
    Batch32: int = 32
    Batch64: int = 64
    Batch128: int = 128

    LRFaster: float = 5e-5
    LRFast: float = 25e-6
    LR: float = 1e-6
    LRSlow: float = 25e-7
    LRSlower: float = 1e-7

    Seed: int = 42
    Device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CheckPoint: str = "./saved_models"
    SavePath: str = "./predictions"


@dataclass
class Optimizer:
    Adam: str = "Adam"
    AdamP: str = "AdamP"
    AdamW: str = "AdamW"
    SGD: str = "SGD"
    Momentum: str = "Momentum"
    CosineAnnealing: str = "CosineScheduler"
    LambdaLR = "LambdaLR"


@dataclass
class Loss:
    CE: str = "crossentropyloss"
    LS: str = "labelsmoothingLoss"


@dataclass
class PreProcessType:
    Base: str = "Base"  # No preprocessing
    ES: str = (
        "EntitySeparation"  # Entity Separation, method as baseline of boostcamp itself
    )
    ESP: str = "ESPositionEmbedding"  # Entity Separation with Position Embedding, add scalar for each values in entities
    EM: str = "EntityMarker"  # Entity Marking
    EMSP: str = "EntityMarkerSeparationPositionEmbedding"


@dataclass
class ModelType:
    VanillaBert: str = "VanillaBert"
    VanillaBert_v2: str = "VanillaBert_v2"
    Base: str = "BertModel"
    SequenceClf: str = "BertForSequenceClassification"
    KoELECTRAv3: str = "KoELECTRAv3"
    KoBert: str = "monologg/kobert"

    XLMSequenceClf: str = "XLMSequenceClf"
    XLMBase: str = "XLMBase"


@dataclass
class PreTrainedType:
    MultiLingual: str = "bert-base-multilingual-cased"
    BaseUncased: str = "bert-base-uncased"
    KoELECTRAv3: str = "monologg/koelectra-base-v3-discriminator"
    KoBert: str = "monologg/kobert"
    XLMRoberta: str = "roberta-base"
