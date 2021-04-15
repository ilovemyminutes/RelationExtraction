from transformers import BertTokenizer
from config import TokenizationType


def load_tokenizer(type: str = TokenizationType.Base):
    """사전 학습된 tokenizer를 불러오는 함수
    Args
    ---
    - type(str): 불러올 tokenizer 타입을 설정. (default: 'Base')

    Return
    ---
    - tokenizer(BertTokenizer): 사전 학습된 tokenizer
    """
    if type == TokenizationType.Base:
        tokenizer = BertTokenizer.from_pretrained(TokenizationType.Base)
    else:
        raise NotImplementedError
    return tokenizer
