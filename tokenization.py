from dataclasses import dataclass
from transformers import BertTokenizer, AutoTokenizer
from config import PreProcessType, PreTrainedType


@dataclass
class SpecialToken:
    # Basic Special Tokens for BERT
    SEP: str = "[SEP]"
    CLS: str = "[CLS]"

    # 무엇(entity1)은 무엇(entity2)과 어떤 관계이다.
    E1Open: str = "[E1]"
    E1Close: str = "[/E1]"
    E2Open: str = "[E2]"
    E2Close: str = "[/E2]"

    # 무엇(sub)은 무엇(obj)과 어떤 관계이다.
    SUBOpen: str = "[SUB]"
    SUBClose: str = "[/SUB]"
    OBJOpen: str = "[OBJ]"
    OBJClose: str = "[/OBJ]"


def load_tokenizer(type: str = PreProcessType.Base):
    """사전 학습된 tokenizer를 불러오는 함수
    Args
    ---
    - type(str): 불러올 tokenizer 타입을 설정. (default: 'Base')

    Return
    ---
    - tokenizer(BertTokenizer): 사전 학습된 tokenizer
    """
    print("Load Tokenizer...", end="\t")
    if type in [PreProcessType.Base, PreProcessType.ES, PreProcessType.ESP]:
        tokenizer = BertTokenizer.from_pretrained(PreTrainedType.MultiLingual)
    else:
        raise NotImplementedError
    print("done!")
    return tokenizer
