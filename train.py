import torch
from tokenizers import load_tokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from models import load_model
from dataset import REDataset, apply_tokenization, load_data
from config import ModelType, Config, PreTrainedType, TokenizationType, TrainArgs
from evaluation import compute_metrics


def train(
    data_root: str = Config.Train,
    model_type: str = ModelType.SequenceClf,
    pretrained_type: str = PreTrainedType.BertMultiLingual,
    tokenization_type: str = TokenizationType.Base,
    num_classes: int = Config.NumClasses,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(model_type, pretrained_type, num_classes)
    model.to(device)
    tokenizer = load_tokenizer(type=tokenization_type)

    # TODO: 우선 이렇게 작성해본 뒤에, load_data 내에 apply_tokenization를 포함시키는 것이 좋을 것 같으면 수정하자.
    dataset_raw, labels = load_data(path=data_root)
    dataset_tokenized = apply_tokenization(
        dataset=dataset_raw, tokenizer=tokenizer, method=TokenizationType.Base
    )
    dataset = REDataset(tokenized_dataset=dataset_tokenized, labels=labels)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(**TrainArgs.Base)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # train model
    trainer.train()


if __name__ == "__main__":
    train()
    pass
