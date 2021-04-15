import os
from dataclasses import dataclass

from tokenizers.implementations.base_tokenizer import BaseTokenizer

LABELTYPE = "./input/data/label_type.pkl"
TRAIN = './input/data/train/train.tsv'
TEST = './input/data/test/test.tsv'
LABEL = './input/data/label_type.pkl'

DOT = '.'

@dataclass
class Config:
    '''notebooks 디렉토리의 주피터 환경에서 아래의 configuration을 활용할 수 있도록 구성
    DOT + SOMETHING: '../something/something' <- 디렉토리 경로를 바꿔주게 됨
    '''
    LabelType: str = LABELTYPE if os.path.isfile(LABELTYPE) else DOT + LABELTYPE
    Train: str = TRAIN if os.path.isfile(TRAIN) else DOT + TRAIN
    Test: str = TEST if os.path.isfile(TEST) else DOT + TEST
    Label: str = LABEL if os.path.isfile(LABEL) else DOT + LABEL
    BertMultiLingual: str = "bert-base-multilingual-cased"
    BaseTokenizer: str='BaseTokenizer'