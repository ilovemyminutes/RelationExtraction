import os
from dataclasses import dataclass


@dataclass
class Config:
    LabelType: str = "./input/data/label_type.pkl" 
    Train: str = './input/data/train/train.tsv' if os.path.isfile('./input/data/train/train.tsv') else '../input/data/train/train.tsv'
    Test: str = './input/data/test/test.tsv' if os.path.isfile('./input/data/test/test.tsv') else '../input/data/test/test.tsv'
