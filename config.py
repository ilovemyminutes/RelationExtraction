import os
from dataclasses import dataclass


def __clarity(path):
    """notebooks 디렉토리 내에서 주피터 활용시 경로가 맞지 않는 문제 해결을 위한 (임시적) 함수"""
    DOT = '.'
    return path if os.path.isfile(path) else DOT + path

@dataclass
class Config:
    LabelType: str = "./input/data/label_type.pkl" if os.path.isfile("./input/data/label_type.pkl") else "../input/data/label_type.pkl"
    Train: str = './input/data/train/train.tsv' if os.path.isfile('./input/data/train/train.tsv') else '../input/data/train/train.tsv'
    Test: str = './input/data/test/test.tsv' if os.path.isfile('./input/data/test/test.tsv') else '../input/data/test/test.tsv'
    BERTMultiLingual: str = "bert-base-multilingual-cased"
    WikiSmall: str = './my_data/wiki_20190620_small.txt' if os.path.isfile('./input/data/test/test.tsv') else '../input/data/test/test.tsv'