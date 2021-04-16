from transformers import BertModel, BertConfig, BertForSequenceClassification
from config import ModelType, Config, ModelType, PreTrainedType


def load_model(
    model_type: str = ModelType.SequenceClf,
    pretrained_type: str = PreTrainedType.BertMultiLingual,
    num_classes: int = Config.NumClasses,
):
    print("Load Model...", end="\t")
    bert_config = BertConfig.from_pretrained(pretrained_type)
    bert_config.num_labels = num_classes

    if model_type == ModelType.Base:
        model = BertModel.from_pretrained(
            pretrained_type, config=bert_config
        )
    elif model_type == ModelType.SequenceClf:
        model = BertForSequenceClassification.from_pretrained(
            pretrained_type, config=bert_config
        )
    else:
        raise NotImplementedError()
    print("done!")
    return model
