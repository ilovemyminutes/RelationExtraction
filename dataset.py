import random
from typing import Tuple, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers.utils import logging

logger = logging.get_logger(__name__)
from config import Config, PreProcessType
from utils import load_pickle
from tokenization import load_tokenizer, tokenize
from preprocessing import preprocess_text


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


class REDataset(Dataset):
    def __init__(
        self,
        root: str = Config.Train,
        preprocess_type: str = PreProcessType.EM,
        device: str = Config.Device,
    ):
        self.data = self._load_preprocessed_data(root, preprocess_type=preprocess_type)
        self.inputs = self.data["input"].tolist()
        self.labels = self.data["label"].tolist()
        self.tokenizer = load_tokenizer(type=preprocess_type)
        self.preprocess_type = preprocess_type
        self.device = device

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """모델에 입력할 데이터 생성시, device 상황에 따라 CPU 또는 GPU에 할당한 채로 return"""
        sentence = tokenize(self.inputs[idx], self.tokenizer, self.preprocess_type)

        for key in sentence.keys():
            sentence[key] = sentence[key].to(self.device)

        label = torch.as_tensor(self.labels[idx]).to(self.device)  # device 할당
        return sentence, label

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _load_preprocessed_data(root: str, preprocess_type: str) -> pd.DataFrame:
        enc = LabelEncoder()
        print("Load raw data...", end="\t")
        raw = pd.read_csv(root, sep="\t", header=None)
        raw.columns = COLUMNS
        raw = raw.drop("id", axis=1)
        raw["label"] = raw["label"].apply(lambda x: enc.transform(x))
        print(f"apply preprocess '{preprocess_type}'...", end="\t")
        data = preprocess_text(raw, method=preprocess_type)
        print("done!")
        return data


# TODO: K-Fold


def split_train_test_loader(
    dataset: Dataset,
    test_size: float = 0.2,
    train_batch_size: int = 32,
    test_batch_size: int = 512,
    drop_last: bool = True,
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
        raise ValueError("test_size should be between 0 and 1.")

    num_samples = len(dataset)
    indices = [i for i in range(num_samples)]

    if shuffle:
        random.shuffle(indices)

    num_test = int(test_size * num_samples)

    # train loader
    train_indices = indices[num_test:]
    train_sampler = SubsetRandomSampler(train_indices)
    if drop_last:
        train_loader = DataLoader(
            dataset, sampler=train_sampler, batch_size=train_batch_size, drop_last=True
        )
    else:
        train_loader = DataLoader(
            dataset, sampler=train_sampler, batch_size=train_batch_size, drop_last=False
        )

    # test loader
    test_indices = indices[:num_test]
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(
        dataset, sampler=test_sampler, batch_size=test_batch_size, drop_last=False
    )

    return train_loader, test_loader


# userd for EDA mainly
def load_data(
    path: str, drop_id: bool = True, encode_label: bool = True
) -> pd.DataFrame:
    """데이터를 불러오는 함수로, EDA를 위해 활용

    Args:
        path (str): 데이터 경로
        drop_id (bool, optional): ID 컬럼을 제외할 지 여부를 설정. Defaults to True.
        encode_label (bool, optional): 레이블을 인코딩할 지 여부를 설정. Defaults to True.

    Returns:
        pd.DataFrame: 적어도 relation state, entity1/entity2 텍스트와 위치, 레이블이 포함된 데이터프레임
    """
    data = pd.read_csv(path, sep="\t", header=None, names=COLUMNS)

    # test data have no labels
    if path == Config.Test:
        data.drop("label", axis=1, inplace=True)

    # drop 'id' column
    if drop_id:
        data.drop("id", axis=1, inplace=True)

    # encode label from string to integer
    if encode_label:
        enc = LabelEncoder()
        data["label"] = data["label"].apply(lambda x: enc.transform(x))

    return data


class LabelEncoder:
    def __init__(self, meta_root: str = Config.Label):
        self.encoder = load_pickle(meta_root)
        self.decoder = {j: i for j, i in self.encoder.items()}
        self.encoder["blind"] = 42
        self.decoder[42] = "blind"

    def transform(self, x):
        return self.encoder[x]

    def inverse_transform(self, x):
        return self.decoder[x]


# just for debug
if __name__ == "__main__":
    config_dataset = dict(root=Config.Train, preprocess_type=PreProcessType.EM)
    dataset = REDataset(**config_dataset)
    # train_loader, valid_loader = split_train_test_loader(dataset)
