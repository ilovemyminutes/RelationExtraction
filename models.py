import torch
from torch import nn
from transformers import BertModel, BertConfig, BertForSequenceClassification, ElectraModel, ElectraTokenizer
from config import ModelType, Config, ModelType, PreTrainedType
from dataset import REDataset, split_train_test_loader


def load_model(
    model_type: str = ModelType.SequenceClf,
    pretrained_type: str = PreTrainedType.MultiLingual,
    num_classes: int = Config.NumClasses,
    load_state_dict: str = None,
    pooler_idx: int = 0,  # get last hidden state from CLS
    dropout: float = 0.5
):
    print("Load Model...", end="\t")
    # make BERT configuration
    

    # load pre-trained model
    if model_type == ModelType.Base:
        bert_config = BertConfig.from_pretrained(pretrained_type)
        bert_config.num_labels = num_classes
        model = BertModel.from_pretrained(pretrained_type, config=bert_config)
    elif model_type == ModelType.SequenceClf:
        bert_config = BertConfig.from_pretrained(pretrained_type)
        bert_config.num_labels = num_classes
        model = BertForSequenceClassification.from_pretrained(
            pretrained_type, config=bert_config
        )
    elif model_type == ModelType.VanillaBert:
        bert_config = BertConfig.from_pretrained(pretrained_type)
        bert_config.num_labels = num_classes
        model = VanillaBert(
            model_type=ModelType.SequenceClf,
            pretrained_type=pretrained_type,
            num_labels=num_classes,
            pooler_idx=pooler_idx,
            dropout=dropout
        )
    elif model_type == ModelType.VanillaBert_v2:
        bert_config = BertConfig.from_pretrained(pretrained_type)
        bert_config.num_labels = num_classes
        model = VanillaBert_v2(
            model_type=ModelType.SequenceClf,
            pretrained_type=pretrained_type,
            num_labels=num_classes,
            pooler_idx=pooler_idx
        )
    elif model_type == ModelType.KoELECTRAv3:
        model = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")

    else:
        raise NotImplementedError()

    if load_state_dict is not None:
        model.load_state_dict(torch.load(load_state_dict))
        print(f"Loaded pretrained weights from {load_state_dict}", end="\t")

    print("done!")
    return model


class VanillaBert_v2(nn.Module):
    def __init__(
        self,
        model_type: str = ModelType.SequenceClf,  # BertForSequenceClassification
        pretrained_type: str = PreTrainedType.MultiLingual,  # bert-base-multilingual-cased
        num_labels: int = Config.NumClasses,  # 42
        pooler_idx: int = 0
    ):
        super(VanillaBert_v2, self).__init__()
        bert = self.load_bert(
            model_type=model_type,
            pretrained_type=pretrained_type,
            num_labels=num_labels
        )
        self.backbone = bert.bert
        self.dropout = bert.dropout
        self.clf = bert.classifier
        self.idx = 0 if pooler_idx == 0 else pooler_idx

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.backbone(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        x = x.last_hidden_state[:, self.idx, :]
        x = self.dropout(x)
        output = self.clf(x)
        return output

    @staticmethod
    def load_bert(model_type, pretrained_type, num_labels):
        config = BertConfig.from_pretrained(pretrained_type)
        config.num_labels = num_labels
        if model_type == ModelType.SequenceClf:
            model = BertForSequenceClassification.from_pretrained(pretrained_type, config=config)
        else:
            raise NotImplementedError()

        return model


class VanillaBert(nn.Module):
    def __init__(
        self,
        model_type: str = ModelType.SequenceClf,  # BertForSequenceClassification
        pretrained_type: str = PreTrainedType.MultiLingual,  # bert-base-multilingual-cased
        num_labels: int = Config.NumClasses,  # 42
        pooler_idx: int = 0,
        dropout: float = 0.5,
    ):
        super(VanillaBert, self).__init__()
        # BERT로부터 얻은 128(=max_length)개 hidden state 중 몇 번째를 활용할 지 결정. Default - 0(CLS 토큰의 인덱스)
        self.idx = 0 if pooler_idx == 0 else pooler_idx
        self.backbone = self.load_bert(
            model_type=model_type,
            pretrained_type=pretrained_type,
        )
        self.layernorm = nn.LayerNorm(768)  # 768: output length of backbone, BERT
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=768, out_features=num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.backbone(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        # backbone으로부터 얻은 128(토큰 수)개 hidden state 중 어떤 것을 활용할 지 결정. Default - 0(CLS 토큰)
        x = x.last_hidden_state[:, self.idx, :]
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.relu(x)
        output = self.linear(x)
        return output

    @staticmethod
    def load_bert(model_type, pretrained_type):
        if model_type == ModelType.SequenceClf:
            model = BertForSequenceClassification.from_pretrained(pretrained_type)
            model = model.bert  # 마지막 레이어을 제외한 BERT 아키텍쳐만을 backbone으로 사용
        else:
            raise NotImplementedError()

        return model


if __name__ == "__main__":
    dataset = REDataset()
    train_loader, valid_loader = split_train_test_loader(dataset)
    model = VanillaBert().cuda()
    for sents, labels in train_loader:
        break
    model(**sents)
