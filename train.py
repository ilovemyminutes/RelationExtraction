from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from evaluation import evaluate
from transformers import Trainer, TrainingArguments
from models import load_model
from dataset import REDataset, get_train_test_loader
from optimizers import get_optimizer, get_scheduler
from criterions import get_criterion
from config import ModelType, Config, Optimizer, PreTrainedType, TokenizationType, Loss


VALID_CYCLE = 100


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
    lr_scheduler: str = Optimizer.CosineScheduler,
    device: str = Config.Device,
):
    # load data
    dataset = REDataset(
        root=data_root, tokenization_type=tokenization_type, device=device
    )
    train_loader, valid_loader = get_train_test_loader(dataset)

    # load model
    model = load_model(model_type, pretrained_type, num_classes)
    model.to(device)
    model.train()

    # load object func, optimizer
    criterion = get_criterion(type=loss_type)
    optimizer = get_optimizer(model=model, type=optim_type, lr=lr)
    if lr_scheduler is not None:
        scheduler = get_scheduler(type=lr_scheduler)

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        # ACC, RECALL, PRECISION, F1
        pred_list = []
        true_list = []
        total_loss = 0  # CE Loss

        for idx, (sentences, labels) in tqdm(enumerate(train_loader), desc="[Train]"):
            outputs = model(**sentences).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                scheduler.step()

            _, preds = torch.max(outputs, dim=1)
            preds = preds.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            pred_list.append(preds)
            true_list.append(labels)

            pred_arr = np.hstack(pred_list)
            true_arr = np.hstack(true_list)

            # evaluation phase
            train_eval = evaluate(y_true=true_arr, y_pred=pred_arr)  # ACC, F1, PRC, REC
            train_loss = total_loss / len(true_arr)

            if idx != 0 and idx % VALID_CYCLE == 0:
                valid_eval, valid_loss = validate(
                    model=model, valid_loader=valid_loader, criterion=criterion
                )


def validate(model, valid_loader, criterion):
    pred_list = []
    true_list = []
    total_loss = 0

    with torch.no_grad():
        model.eval()
        for sentences, labels in tqdm(valid_loader, desc="[Valid]"):
            outputs = model(**sentences).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            preds = preds.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            pred_list.append(preds)
            true_list.append(labels)

        pred_arr = np.hstack(pred_list)
        true_arr = np.hstack(true_list)

        # evaluation phase
        valid_eval = evaluate(y_true=true_arr, y_pred=pred_arr)  # ACC, F1, PRC, REC
        valid_loss = total_loss / len(true_arr)
        model.train()

    return valid_eval, valid_loss


if __name__ == "__main__":
    train()
    pass
