from tqdm import tqdm
from torch import nn, optim
from evaluation import compute_metrics
from transformers import Trainer, TrainingArguments
from models import load_model
from dataset import REDataset, get_train_test_loader
from config import ModelType, Config, PreTrainedType, TokenizationType
from optims import get_optim, get_scheduler


def train(
    data_root: str = Config.Train,
    model_type: str = ModelType.SequenceClf,
    epochs: int = 1, 
    pretrained_type: str = PreTrainedType.BertMultiLingual,
    tokenization_type: str = TokenizationType.Base,
    num_classes: int = Config.NumClasses,
    optim_type: str = 
    device: str = Config.Device
):
    # load data
    dataset = REDataset(root=data_root, tokenization_type=tokenization_type, device=device)
    train_loader, valid_loader = get_train_test_loader(dataset)

    # load model
    model = load_model(model_type, pretrained_type, num_classes)
    model.to(device)
    model.train()

    # load object func, optimizer
    criterion = nn.CrossEntropyLoss()
    optim = get_optim(model, optim_type=, lr)
    

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')

        for sents, labels in tqdm(train_loader, desc='Train'):
            outputs = model(**sents).logits
            loss = cri
    

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
