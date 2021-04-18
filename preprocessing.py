import pandas as pd
from tokenization import SpecialToken as ST
from config import PreProcessType


def preprocess_text(data: pd.DataFrame, method: str=PreProcessType.Base):
    """Preprocessing 방법에 따라 텍스트를 전처리하는 함수.
    data를 전처리한 후 토크나이저는 'input' 컬럼에 대해 토크나이징하게 됨

    Args:
        data (pd.DataFrame): raw 데이터
        method (str, optional): 전처리 방식을 설정. Defaults to PreProcessType.Base.

    Returns:
        pd.DataFrame: [description]
    """    
    if method == PreProcessType.Base:
        data['input'] = data['relation_state']
        return data

    elif method in [PreProcessType.ES, PreProcessType.ESP]:
        if 'e1' not in data.columns or 'e2' not in data.columns:
            raise ValueError(f"'e1' or 'e2' column not in input data")
        data['input'] = data['e1'] + ST.SEP + data['e2'] + ST.SEP + data['relation_state']
        return data

    elif method == PreProcessType.EM:
        pass

    else:
        raise NotImplementedError(f"There's no method for '{method}'")