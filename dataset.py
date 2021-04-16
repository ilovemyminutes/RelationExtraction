import random
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers.utils import logging
logger = logging.get_logger(__name__)
from config import Config, TokenizationType
from utils import load_pickle
from tokenization import load_tokenizer


# TODO: K-Fold

def get_train_test_loader(
    dataset: Dataset,
    train_batch_size: int = 32,
    test_batch_size: int = 512,
    drop_last: bool = True,
    test_size: float = 0.2,
    shuffle: bool = True,
):
    """데이터셋을 입력 받아 train, test DataLoader를 생성하는 함수

    Args:
        dataset (Dataset): 
        train_batch_size (int, optional): 학습용 DataLoader의 배치 사이즈. Defaults to 64.
        valid_batch_size (int, optional): 검증용 DataLaoder의 배치 사이즈. Defaults to 512.
        drop_last (boot, optional): Train DataLoader의 마지막 배치를 버릴지 여부. Defaults to True.
        test_size (float, optional): 얼만큼의 비율로 데이터를 나눌지 결정. Defaults to 0.2.
        shuffle (bool, optional):
            데이터 분리 과정에서 셔플을 진행할 지 여부
            NOTE. 분리 이후 생성된 DataLoader는 shuffle 여부에 관계 없이 random iteration

    Returns:
        train_loader (DataLoader): 학습용 DataLoader
        test_loader (DataLoader): 검증용 DataLoader
    """
    if test_size == 0 or test_size > 1:
        raise ValueError('test_size should be between 0 and 1.')
        
    num_samples = len(dataset)
    indices = [i for i in range(num_samples)]

    if shuffle:
        random.shuffle(indices)

    num_test = int(test_size * num_samples)

    # train loader
    train_indices = indices[num_test:]
    train_sampler = SubsetRandomSampler(train_indices)
    if drop_last:
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=train_batch_size, drop_last=True)
    else:
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=train_batch_size, drop_last=False)

    # test loader
    test_indices = indices[:num_test]
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=test_batch_size, drop_last=False)

    return train_loader, test_loader


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
        "label",
    ]

    def __init__(
        self, root: str = Config.Train, tokenization_type: str = TokenizationType.Base
    ):
        self.tokenizer = load_tokenizer(type=tokenization_type)
        self.enc = LabelEncoder()
        raw = self._load_raw(root)
        self.sentences = self._tokenize(raw)
        self.labels = raw["label"].tolist()

    def __getitem__(self, idx) -> Tuple[dict, torch.Tensor]:
        sentence = {
            key: torch.as_tensor(val[idx]) for key, val in self.sentences.items()
        }
        label = torch.as_tensor(self.labels[idx])
        return sentence, label

    def __len__(self):
        return len(self.labels)

    def _load_raw(self, root):
        print("Load raw data...", end="\t")
        raw = pd.read_csv(root, sep="\t", header=None)
        raw.columns = self.COLUMNS
        raw = raw.drop("id", axis=1)
        raw["label"] = raw["label"].apply(lambda x: self.enc.transform(x))
        print("done!")
        return raw

    def _tokenize(self, data):
        print("Apply Tokenization...", end="\t")
        data_tokenized = self.tokenizer(
            data["relation_state"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
        )
        print("done!")
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
if __name__ == "__main__":
    config_dataset = dict(root=Config.Train, tokenization_type=TokenizationType.Base)
    dataset = REDataset(**config_dataset)
    train_loader, valid_loader = get_train_test_loader(dataset)
