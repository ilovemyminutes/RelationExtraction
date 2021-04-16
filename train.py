import torch
from tokenization import load_tokenizer
from transformers import Trainer, TrainingArguments
from models import load_model
from dataset import REDataset, get_train_test_loader
from config import ModelType, Config, PreTrainedType, TokenizationType, TrainArgs
from evaluation import compute_metrics


def train(
    data_root: str = Config.Train,
    model_type: str = ModelType.SequenceClf,
    pretrained_type: str = PreTrainedType.BertMultiLingual,
    tokenization_type: str = TokenizationType.Base,
    num_classes: int = Config.NumClasses,
    device: str = Config.Device
):
    model = load_model(model_type, pretrained_type, num_classes).to(device)
    dataset = REDataset(root=data_root, tokenization_type=tokenization_type)
    train_loader, valid_loader = get_train_test_loader(dataset)
    

    training_args = TrainingArguments(**TrainArgs.Base)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # data_collator=data_collator, # TODO: Sequence Classification Task에 맞는 Data Collator를 구축
        compute_metrics=compute_metrics
    )
    # train model
    trainer.train()
    trainer.evaluate(eval_dataset=dataset)


if __name__ == "__main__":
    train()
    pass
