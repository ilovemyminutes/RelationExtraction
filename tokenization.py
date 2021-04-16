from transformers import BertTokenizer
from config import TokenizationType, PreTrainedType


def load_tokenizer(type: str = TokenizationType.Base):
    """사전 학습된 tokenizer를 불러오는 함수
    Args
    ---
    - type(str): 불러올 tokenizer 타입을 설정. (default: 'Base')

    Return
    ---
    - tokenizer(BertTokenizer): 사전 학습된 tokenizer
    """
    print("Load Tokenizer...", end="\t")
    if type == TokenizationType.Base:
        tokenizer = BertTokenizer.from_pretrained(PreTrainedType.BertMultiLingual)
    else:
        raise NotImplementedError
    print("done!")
    return tokenizer
