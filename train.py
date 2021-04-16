from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from evaluation import evaluate
from transformers import Trainer, TrainingArguments
from models import load_model
from dataset import REDataset, get_train_test_loader
from optims import get_optimizer, get_scheduler
from criterions import get_criterion
from config import ModelType, Config, Optimizer, PreTrainedType, TokenizationType, Loss


def train(
    data_root: str = Config.Train,
    model_type: str = ModelType.SequenceClf,
    epochs: int = 1, 
    pretrained_type: str = PreTrainedType.BertMultiLingual,
    tokenization_type: str = TokenizationType.Base,
    num_classes: int = Config.NumClasses,
    loss_type: str = Loss.CE,
    optim_type: str = Optimizer.Adam,
    lr: float = Config.LR,
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
    criterion = get_criterion(type=loss_type)
    optimizer = get_optimizer(model, optim_type=, lr=lr)
    

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')

        # ACC, RECALL, PRECISION, F1
        pred_list = []
        true_list = []

        # CE Loss
        total_loss = 0
        num_samples = 0

        for sents, labels in tqdm(train_loader, desc='Train'):
            outputs = model(**sents).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_samples += len(labels)

            _, preds = torch.max(outputs, dim=1)
            preds = preds.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            pred_list.append(preds)
            true_list.append(labels)
        
        pred_arr = np.hstack(pred_list)
        true_arr = np.hstack(true_list)

        # Calculate metrics
        precision, recal, f1, _ = precision_recall_fscore_support(y_true=true_arr, y_pred=pred_arr, average='macro')
        accuracy = accuracy_score(y_true=true_arr, y_pred=pred_arr)
        train_loss = total_loss / num_samples

    

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
