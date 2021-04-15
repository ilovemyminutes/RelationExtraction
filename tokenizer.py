from transformers import BertTokenizer
from config import Config


def load_tokenizer(type: str = Config.BaseTokenizer):
    """사전 학습된 tokenizer를 불러오는 함수
    Args
    ---
    - type(str): 불러올 tokenizer 타입을 설정. (default: 'BaseTokenizer')

    Return
    ---
    - tokenizer(BertTokenizer): 사전 학습된 tokenizer
    """
    if type == Config.BaseTokenizer:
        tokenizer = BertTokenizer.from_pretrained(Config.BertMultiLingual)
    else:
        raise NotImplementedError
    return tokenizer
