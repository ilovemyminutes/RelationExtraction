import os
import pickle
import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from transformers.utils import logging
logger = logging.get_logger(__name__)
from config import Config
from utils import load_pickle


COLUMNS = ['id', 'relation_state', 'e1', 'e1_start', 'e1_end', 'e2', 'e2_start', 'e2_end', 'label']

def load_data(path: str, drop_id: bool=True, encode_label: bool=True):
    data = pd.read_csv(path, sep='\t', header=None, names=COLUMNS)

    if path == Config.Test: # test data have no labels
        data.drop('label', axis=1, inplace=True)
    if drop_id: # drop 'id' column
        data.drop('id', axis=1, inplace=True)
    if encode_label and path != Config.Test: # encode label from string to integer
        enc = LabelEncoder()
        data['label'] = data['label'].apply(lambda x: enc.transform(x))
    return data

# Dataset 구성.
class REDataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == "blind":
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame(
        {
            "sentence": dataset[1],
            "entity_01": dataset[2],
            "entity_02": dataset[5],
            "label": label,
        }
    )
    return out_dataset


class LabelEncoder:
    def __init__(self, meta_root: str=Config.Label):
        self.encoder = load_pickle(Config.Label)
        self.decoder = {j:i for j, i in self.encoder.items()}

    def transform(self, x):
        return self.encoder[x]
        
    def inverse_transform(self, x):
        return self.decoder[x]

# def load_test_dataset(dataset_dir, tokenizer):
#     test_dataset = load_data(dataset_dir)
#     test_label = test_dataset["label"].values
#     # tokenizing dataset
#     tokenized_test = tokenized_dataset(test_dataset, tokenizer)
#     return tokenized_test, test_label