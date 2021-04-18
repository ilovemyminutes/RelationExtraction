import torch
from torch import nn
from transformers import BertModel, BertConfig, BertForSequenceClassification
from config import ModelType, Config, ModelType, PreTrainedType
from dataset import REDataset, split_train_test_loader


def load_model(
    model_type: str = ModelType.SequenceClf,
    pretrained_type: str = PreTrainedType.BertMultiLingual,
    num_classes: int = Config.NumClasses,
    load_state_dict: str = None,
    pooler_idx: int = 0,  # get last hidden state from CLS
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
    elif model_type == ModelType.VanillaBert:
        model = VanillaBert(model_type=ModelType.SequenceClf, pretrained_type=pretrained_type, num_labels=num_classes, pooler_idx=pooler_idx)
    else:
        raise NotImplementedError()

    if load_state_dict is not None:
        model.load_state_dict(torch.load(load_state_dict))
        print(f"Loaded pretrained weights from {load_state_dict}", end="\t")

    print("done!")
    return model


class VanillaBert(nn.Module):
    def __init__(
        self,
        model_type: str,
        pretrained_type: str,
        num_labels: int = Config.NumClasses,
        pooler_idx: int = 0,
    ):
        super(VanillaBert, self).__init__()
        # idx: index of hidden state to extract from output hidden states. It is CLS hidden states for index 0.
        self.idx = 0 if pooler_idx in ["cls", 0] else pooler_idx
        self.backbone = self.load_bert(
            model_type=model_type,
            pretrained_type=pretrained_type,
            num_labels=num_labels,
        )
        self.layernorm = nn.LayerNorm(768)  # 768: output length of BERT, or backbone
        self.dropout = nn.Dropout(p=0.8)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=768, out_features=num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.backbone(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        x = x.last_hidden_state[:, self.idx, :]
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.relu(x)
        output = self.linear(x)
        return output

    @staticmethod
    def load_bert(model_type, pretrained_type, num_labels):
        bert_config = BertConfig.from_pretrained(pretrained_type)
        bert_config.num_labels = num_labels

        if model_type == ModelType.SequenceClf:
            model = BertForSequenceClassification.from_pretrained(
                pretrained_type, config=bert_config
            )
            model = model.bert

        elif model_type == ModelType.Base:
            raise NotImplementedError()

        return model


if __name__ == "__main__":
    dataset = REDataset()
    train_loader, valid_loader = split_train_test_loader(dataset)
    model = VanillaBert().cuda()
    for sents, labels in train_loader:
        break
    model(**sents)
