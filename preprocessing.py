import pandas as pd
from tokenization import SpecialToken as ST
from config import PreProcessType

ENTITY_COLS = ["e1_start", "e1_end", "e2_start", "e2_end"]
POS = ["NNG", "NNP", "NNB", "NNBC", "NR", "NP", "VV", "VA", "XR"]


def preprocess_text(data: pd.DataFrame, method: str = PreProcessType.Base):
    """Preprocessing 방법에 따라 텍스트를 전처리하는 함수.
    텍스트 전처리 결과는 'input' 컬럼으로 추가됨
    이후 토크나이저는 'input' 컬럼에 대해 토크나이징하도록 설정!

    Args:
        data (pd.DataFrame): raw 데이터
        method (str, optional): 전처리 방식을 설정. Defaults to PreProcessType.Base.

    Returns:
        pd.DataFrame: 전처리 타입에 맞게 처리된 데이터프레임
    """
    if method == PreProcessType.Base:
        data["input"] = data["relation_state"]
        return data

    # 기존 제시문 -> [개체1][SEP][개체2][SEP][기존 제시문]
    elif method in [PreProcessType.ES, PreProcessType.ESP]:
        data["input"] = (
            data["e1"] + ST.SEP + data["e2"] + ST.SEP + data["relation_state"]
        )
    
    # [개체 각각 스페셜토큰이 부착된 제시문]
    elif method == PreProcessType.EM:
        data["input"] = data.apply(lambda x: attach_entities(x), axis=1)
        return data

    # [개체1][SEP][개체2][SEP][개체 각각 스페셜토큰이 부착된 제시문]
    elif method == PreProcessType.EMSP:
        data["input"] = data.apply(lambda x: attach_entities(x), axis=1)
        data["input"] = (
            data["e1"] + ST.SEP + data["e2"] + ST.SEP + data["input"]
        )
    else:
        raise NotImplementedError(f"There's no method for '{method}'")
    
    return data


def attach_entities(row: pd.Series):
    sentence = row["relation_state"]
    e1_open, e1_close, e2_open, e2_close = row[ENTITY_COLS].values
    output = ""
    if e1_open > e2_open:
        output += sentence[:e2_open]
        output += ST.E2Open
        output += sentence[e2_open : e2_close + 1]
        output += ST.E2Close
        output += sentence[e2_close + 1 : e1_open]
        output += ST.E1Open
        output += sentence[e1_open : e1_close + 1]
        output += ST.E1Close
        output += sentence[e1_close + 1 :]
    else:
        output += sentence[:e1_open]
        output += ST.E1Open
        output += sentence[e1_open : e1_close + 1]
        output += ST.E1Close
        output += sentence[e1_close + 1 : e2_open]
        output += ST.E2Open
        output += sentence[e2_open : e2_close + 1]
        output += ST.E2Close
        output += sentence[e2_close + 1 :]
    return output
