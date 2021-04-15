from transformers import BertModel, BertConfig
from config import ModelType


def load_model(model_type: type):
    if model_type == ModelType.BertMultiLingual:
        bert_config = BertConfig.from_pretrained(ModelType.BertMultiLingual)
        model = BertModel(ModelType.BertMultiLingual, config=bert_config)
    else:
        raise NotImplementedError()
    return model
