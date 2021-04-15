from transformers import BertModel, BertConfig
from config import ModelType


def load_model(type: type):
    if type == ModelType.BertMultiLingual:
        bert_config = BertConfig.from_pretrained(ModelType.BertMultiLingual)
        model = BertModel.from_pretrained(ModelType.BertMultiLingual, config=bert_config)
    else:
        raise NotImplementedError()
    return model
