from tokenizer import load_tokenizer
import torch
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertConfig,
)
from models import load_model
from dataset import REDataset, apply_tokenization, load_data
from config import ModelType, Config, TokenizationType, TrainArgs


def compute_metrics(pred):
    """평가를 위한 metrics function."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }


def train(
    data_root: str=Config.Train,
    model_type: str=ModelType.BertMultiLingual,
    tokenization_type: str=TokenizationType.Base,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model and tokenizer
    model = load_model(model_type=model_type).to(device)
    tokenizer = load_tokenizer(tokenization_type=tokenization_type)

    dataset = load_data(path=data_root)
    labels = dataset['label'].tolist()
    dataset = apply_tokenization(dataset=dataset, tokenizer=tokenizer, method=TokenizationType.Base)
    dataset = REDataset(tokenized_dataset=dataset, labels=labels)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.

    training_args = TrainingArguments(TrainArgs.Base)
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=dataset,  # training dataset
    )

    # train model
    trainer.train()


def main():
    train()


if __name__ == "__main__":
    main()
