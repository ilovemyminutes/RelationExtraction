from dataclasses import dataclass
from transformers import BertTokenizer, AutoTokenizer
from config import PreProcessType, PreTrainedType

# 토큰화 결과 [CLS] 토큰이 가장 앞에 붙게 되기 떄문에
# Entity Mark에 대한 임베딩 값을 조절하는 과정에서 인덱스를 OFFSET만큼 밀어주기 위해 사용
OFFSET = 1
ENTITY_SCORE = 2
SEP_SCORE = 2


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


def tokenize(sentence, tokenizer, type: str=PreProcessType.Base) -> dict:
    outputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
        add_special_tokens=True,
    )
    if type != PreProcessType.Base:
        tokenized = tokenizer.tokenize(sentence)

        if type == PreProcessType.EM:
            # Add embedding value for entity marker tokens([E1], [/E1], [E2], [/E2])
            entity_indices = find_entity_indices(tokenized)
            for open, close in entity_indices.values():
                outputs.token_type_ids[0][
                    OFFSET + open : OFFSET + close + 1
                ] += ENTITY_SCORE

        elif type == PreProcessType.ESP:
            # Add embedding value for separation token([SEP])
            last_sep_idx = fine_sep_indices(tokenized).pop()
            outputs.token_type_ids[0][OFFSET : last_sep_idx + 1] += SEP_SCORE
            return outputs

        elif type == PreProcessType.EMSP:
            entity_indices = find_entity_indices(tokenized)
            for (open, close) in entity_indices.values():
                outputs.token_type_ids[0][
                    OFFSET + open : OFFSET + close + 1
                ] += ENTITY_SCORE

            last_sep_idx = fine_sep_indices(tokenized).pop()
            outputs.token_type_ids[0][OFFSET : last_sep_idx + 1] += SEP_SCORE

    return outputs


def find_entity_indices(tokenized: list) -> dict:
    entity_indices = {
        "e1": (
            tokenized.index(SpecialToken.E1Open),
            tokenized.index(SpecialToken.E1Close),
        ),
        "e2": (
            tokenized.index(SpecialToken.E2Open),
            tokenized.index(SpecialToken.E2Close),
        ),
    }
    return entity_indices


def fine_sep_indices(tokenized: list) -> list:
    sep_indices = [idx for idx, tok in enumerate(tokenized) if tok == SpecialToken.SEP]
    return sep_indices


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
    elif type in [PreProcessType.EM, PreProcessType.EMSP]:
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


if __name__ == '__main__':
    tokenizer = load_tokenizer(PreProcessType.EM)
    sentence = '영국에서 사용되는 스포츠 유틸리티 [E2]자동차[/E2]의 브랜드로는 [E1]랜드로버[/E1](Land Rover)와 지프(Jeep)가 있으며, 이 브랜드들은 자동차의 종류를 일컫는 말로 사용되기도 한다.'
    tokenize(sentence, tokenizer, PreProcessType.EM)