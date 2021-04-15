import os
import pickle
import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from transformers.utils import logging

logger = logging.get_logger(__name__)
from config import Config, TokenizationType
from utils import load_pickle


COLUMNS = [
    "id",
    "relation_state",
    "e1",
    "e1_start",
    "e1_end",
    "e2",
    "e2_start",
    "e2_end",
    "label",
]


def load_data(path: str, drop_id: bool = True, encode_label: bool = True):
    data = pd.read_csv(path, sep="\t", header=None, names=COLUMNS)

    # test data have no labels
    if path == Config.Test:
        data.drop("label", axis=1, inplace=True)

    # drop 'id' column
    if drop_id:
        data.drop("id", axis=1, inplace=True)

    # encode label from string to integer
    if encode_label and path != Config.Test:
        enc = LabelEncoder()
        data["label"] = data["label"].apply(lambda x: enc.transform(x))

    return data


def apply_tokenization(dataset, tokenizer, method: str = TokenizationType.Base):
    if method == TokenizationType.Base:
        tokenized_dataset = tokenizer(
            dataset["relation_state"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100,
            add_special_tokens=True,
        )
    else:
        raise NotImplementedError
    return tokenized_dataset


class REDataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.as_tensor(val[idx])
            for key, val in self.tokenized_dataset.items()
        }
        item["labels"] = torch.as_tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class LabelEncoder:
    def __init__(self, meta_root: str = Config.Label):
        self.encoder = load_pickle(meta_root)
        self.decoder = {j: i for j, i in self.encoder.items()}

    def transform(self, x):
        return self.encoder[x]

    def inverse_transform(self, x):
        return self.decoder[x]
