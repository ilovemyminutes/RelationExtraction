from tqdm import tqdm
import pandas as pd
import torch
from dataclasses import dataclass
from transformers import BertTokenizer, AutoTokenizer
from config import PreProcessType, PreTrainedType

# 토큰화 결과 [CLS] 토큰이 가장 앞에 붙게 되기 떄문에
# Entity Mark에 대한 임베딩 값을 조절하는 과정에서 인덱스를 OFFSET만큼 밀어주기 위해 사용
OFFSET = 1
ENTITY_SCORE = 1
SEP_SCORE = 1
MAX_LENGTH = 128


@dataclass
class SpecialToken:
    # Basic Special Tokens for BERT
    SEP: str = "[SEP]"
    CLS: str = "[CLS]"

    # 무엇(entity)은 무엇(entity)과 어떤 관계이다.(순서 고려 X)
    EOpen: str = "[ENT]"
    EClose: str = "[/ENT]"

    # 무엇(entity1)은 무엇(entity2)과 어떤 관계이다.
    E1Open: str = "[E1]"
    E1Close: str = "[/E1]"
    E2Open: str = "[E2]"
    E2Close: str = "[/E2]"


def load_tokenizer(type: str = PreProcessType.Base):
    """사전 학습된 tokenizer를 불러오는 함수
    Args
    ---
    - type(str): 불러올 tokenizer 타입을 설정. (default: 'Base')

    Return
    ---
    - tokenizer(BertTokenizer): 사전 학습된 tokenizer
    """
    print(f"Load Tokenizer for {type}...", end="\t")
    if type in [PreProcessType.Base, PreProcessType.ES, PreProcessType.ESP]:
        tokenizer = BertTokenizer.from_pretrained(PreTrainedType.MultiLingual)

    # Entity Marker, Entity Marker Separator with Position Embedding
    elif type == PreProcessType.EM or type == PreProcessType.EMSP:
        tokenizer = BertTokenizer.from_pretrained(PreTrainedType.MultiLingual)
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    SpecialToken.E1Open,
                    SpecialToken.E1Close,
                    SpecialToken.E2Open,
                    SpecialToken.E2Close,
                ]
            }
        )
    else:
        raise NotImplementedError
    print("done!")
    return tokenizer


def tokenize(data: pd.DataFrame, tokenizer, type: str=PreProcessType.Base):
    print("Apply Tokenization...", end="\t")
    data_tokenized = tokenizer(
        data["input"].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )
    if type not in  [PreProcessType.Base, PreProcessType.ES]:
        tokenized_decoded = data["input"].apply(lambda x: tokenizer.tokenize(x))

        if type == PreProcessType.EM:
            # entity marker
            entity_intervals = tokenized_decoded.apply(
                lambda x: find_entity_intervals(x)
            ).tolist()
            entity_interval_tensor = make_additional_token_type_ids(
                entity_intervals, data_size=data.shape[0]
            )
            data_tokenized["token_type_ids"] += entity_interval_tensor.long()

        elif type == PreProcessType.EMSP:
            # entity marker
            entity_intervals = tokenized_decoded.apply(
                lambda x: find_entity_intervals(x)
            ).tolist()
            entity_interval_tensor = make_additional_token_type_ids(
                entity_intervals, data_size=data.shape[0], type='entity'
            )

            # entity separation
            sep_intervals = tokenized_decoded.apply(lambda x: find_sep_intervals(x)).tolist()
            sep_interval_tensor = make_additional_token_type_ids(
                sep_intervals, data_size=data.shape[0], type='sep'
            )

            data_tokenized["token_type_ids"] += entity_interval_tensor.long()
            data_tokenized["token_type_ids"] += sep_interval_tensor.long()

        elif type == PreProcessType.ESP:
            # entity separation
            sep_intervals = tokenized_decoded.apply(lambda x: find_sep_intervals(x)).tolist()
            sep_interval_tensor = make_additional_token_type_ids(
                sep_intervals, data_size=data.shape[0], type='sep'
            )
            data_tokenized["token_type_ids"] += sep_interval_tensor.long()

    return data_tokenized


def find_sep_intervals(tokenized: list) -> list:
    sep_indices = tuple(idx for idx, tok in enumerate(tokenized) if tok == SpecialToken.SEP)
    return sep_indices


def find_entity_intervals(tokenized: list) -> dict:
        entity_intervals = [
            (tokenized.index(SpecialToken.E1Open), tokenized.index(SpecialToken.E1Close)),
            (tokenized.index(SpecialToken.E2Open), tokenized.index(SpecialToken.E2Close))
        ]
        return entity_intervals

def make_additional_token_type_ids(intervals: list, data_size: int, type: str='entity'):
        n_rows = data_size
        n_cols = MAX_LENGTH
        additional_token_type_ids = torch.zeros(n_rows, n_cols)

        if type == 'entity':
            for idx, (e1, e2) in tqdm(enumerate(intervals), desc="Update token_type_ids"):
                additional_token_type_ids[idx][OFFSET+e1[0]: OFFSET+e1[1]+1] += ENTITY_SCORE
                additional_token_type_ids[idx][OFFSET+e2[0]: OFFSET+e2[1]+1] += ENTITY_SCORE

        elif type == 'sep':
            for idx, sep in tqdm(enumerate(intervals), desc="Update token_type_ids"):
                last_sep = sep[-1]
                additional_token_type_ids[idx][OFFSET: OFFSET+last_sep+1] += SEP_SCORE
                additional_token_type_ids[idx][OFFSET: OFFSET+last_sep+1] += SEP_SCORE
                
        return additional_token_type_ids



if __name__ == '__main__':
    tokenizer = load_tokenizer(PreProcessType.ES)
    sentence = '영국에서 사용되는 스포츠 유틸리티 [E2]자동차[/E2]의 브랜드로는 [E1]랜드로버[/E1](Land Rover)와 지프(Jeep)가 있으며, 이 브랜드들은 자동차의 종류를 일컫는 말로 사용되기도 한다.'
    tokenize(sentence, tokenizer, PreProcessType.ES)