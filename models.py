import torch
from transformers import BertModel, BertConfig, BertForSequenceClassification
from config import ModelType, Config, ModelType, PreTrainedType


def load_model(
    model_type: str = ModelType.SequenceClf,
    pretrained_type: str = PreTrainedType.BertMultiLingual,
    num_classes: int = Config.NumClasses,
    load_state_dict: str = None
):
    print("Load Model...", end="\t")
    # make BERT configuration
    bert_config = BertConfig.from_pretrained(pretrained_type)
    bert_config.num_labels = num_classes

    # load pre-trained model
    if model_type == ModelType.Base:
        model = BertModel.from_pretrained(pretrained_type, config=bert_config)
    elif model_type == ModelType.SequenceClf:
        model = BertForSequenceClassification.from_pretrained(
            pretrained_type, config=bert_config
        )
    else:
        raise NotImplementedError()

    if load_state_dict is not None:
        model.load_state_dict(torch.load(load_state_dict))
        print(f"Loaded pretrained weights from {load_state_dict}", end='\t')

    print("done!")
    return model
