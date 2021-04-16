import os
import pickle
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset
import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from transformers.utils import logging

logger = logging.get_logger(__name__)
from config import Config, TokenizationType
from utils import load_pickle
from tokenization import load_tokenizer



class REDataset(Dataset):
    COLUMNS = [
    "id",
    "relation_state",
    "e1",
    "e1_start",
    "e1_end",
    "e2",
    "e2_start",
    "e2_end",
    "label"
    ]
    def __init__(self, root: str=Config.Train, tokenization_type: str=TokenizationType.Base):
        self.tokenizer = load_tokenizer(type=tokenization_type)
        self.enc = LabelEncoder()
        raw = self._load_raw(root)
        self.sentences = self._tokenize(raw)
        self.labels = raw['label'].tolist()

    def __getitem__(self, idx):
        sentence = {
            key: torch.as_tensor(val[idx])
            for key, val in self.sentences.items()
        }
        label = torch.as_tensor(self.labels[idx])
        return sentence, label

    def __len__(self):
        return len(self.labels)
    
    def _load_raw(self, root):
        print('Load raw data...', end='\t')
        raw = pd.read_csv(root, sep='\t', header=None)
        raw.columns = self.COLUMNS
        raw = raw.drop('id', axis=1)
        raw['label'] = raw['label'].apply(lambda x: self.enc.transform(x))
        print('done!')
        return raw

    def _tokenize(self, data):
        print('Apply Tokenization...', end='\t')
        data_tokenized = self.tokenizer(
            data["relation_state"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
        )
        print('done!')
        return data_tokenized



class LabelEncoder:
    def __init__(self, meta_root: str = Config.Label):
        self.encoder = load_pickle(meta_root)
        self.decoder = {j: i for j, i in self.encoder.items()}

    def transform(self, x):
        return self.encoder[x]

    def inverse_transform(self, x):
        return self.decoder[x]


# just for debug
if __name__ == '__main__':
    config_load_data = dict(path=Config.Train, drop_id= True, encode_label=True)
    load_data(**config_load_data)