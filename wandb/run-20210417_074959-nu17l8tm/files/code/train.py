import argparse
import os
import warnings
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import wandb
from evaluation import evaluate
from models import load_model
from dataset import REDataset, split_train_test_loader
from optimizers import get_optimizer, get_scheduler
from criterions import get_criterion
from utils import get_timestamp, set_seed, verbose, ckpt_name
from config import ModelType, Config, Optimizer, PreTrainedType, TokenizationType, Loss

warnings.filterwarnings("ignore")

TOTAL_SAMPLES = 9000

def train(
    model_type: str = ModelType.SequenceClf,
    pretrained_type: str = PreTrainedType.BertMultiLingual,
    num_classes: int = Config.NumClasses,
    pooler_idx: int = 0, 
    load_state_dict: str = None,
    data_root: str = Config.Train,
    tokenization_type: str = TokenizationType.Base,
    epochs: int = Config.Epochs,
    valid_size: float = Config.ValidSize,
    train_batch_size: int = Config.Batch32,
    valid_batch_size: int = 512,
    optim_type: str = Optimizer.Adam,
    loss_type: str = Loss.CE,
    lr: float = Config.LR,
    lr_scheduler: str = Optimizer.CosineScheduler,
    device: str = Config.Device,
    seed: int = Config.Seed,
    save_path: str = Config.CheckPoint,
):
    # set seed
    set_seed(seed)

    # load data
    dataset = REDataset(
        root=data_root, tokenization_type=tokenization_type, device=device
    )
    if valid_size == 0:
        is_valid = False # validation flag
        train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    else:
        is_valid = True # validation flag
        train_loader, valid_loader = split_train_test_loader(
            dataset=dataset,
            test_size=valid_size,
            train_batch_size=train_batch_size,
            test_batch_size=valid_batch_size,
        )

    # load model
    model = load_model(model_type, pretrained_type, num_classes, load_state_dict, pooler_idx)
    model.to(device)
    model.train()

    # load criterion, optimizer, scheduler
    criterion = get_criterion(type=loss_type)
    optimizer = get_optimizer(model=model, type=optim_type, lr=lr)
    if lr_scheduler is not None:
        scheduler = get_scheduler(type=lr_scheduler, optimizer=optimizer)
    
    # make checkpoint directory to save model during train
    checkpoint_dir = f"{model_type}_{pretrained_type}_{TIMESTAMP}"
    if checkpoint_dir not in os.listdir(save_path):
        os.mkdir(os.path.join(save_path, checkpoint_dir))
    save_path = os.path.join(save_path, checkpoint_dir)

    # train phase
    best_acc = 0
    best_loss = 999

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        pred_list = []
        true_list = []
        total_loss = 0

        for idx, (sentences, labels) in tqdm(enumerate(train_loader), desc="[Train]"):
            if model_type == ModelType.SequenceClf:
                outputs = model(**sentences).logits
            elif model_type == ModelType.Base:
                outputs = model(**sentences).pooler_output
            else:
                outputs = model(**sentences)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                scheduler.step()

            # stack preds for evaluate
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

            if epoch == 0:
                wandb.log(
                    {
                        f"First EP Train ACC": train_eval["accuracy"],
                        f"First EP Train F1": train_eval["f1"],
                        # f"First EP Train PRC": train_eval["precision"],
                        # f"First EP Train REC": train_eval["recall"],
                        f"First EP Train Loss": train_loss,
                    }
                ) 

            if (is_valid) and (idx != 0) and (idx % VALID_CYCLE == 0):
                valid_eval, valid_loss = validate(
                    model=model, model_type=model_type, valid_loader=valid_loader, criterion=criterion
                )
                verbose(phase="Valid", eval=valid_eval, loss=valid_loss)
                verbose(phase="Train", eval=train_eval, loss=train_loss)

                if epoch == 0:
                    wandb.log(
                        {
                            f"First EP Valid        ACC": train_eval["accuracy"],
                            f"First EP Valid F1": train_eval["f1"],
                            # f"First EP Valid PRC": train_eval["precision"],
                            # f"First EP Valid REC": train_eval["recall"],
                            f"First EP Valid Loss": train_loss,
                        }
                    )

        
        # logs for one epoch in total
        if is_valid:
            wandb.log(
                {
                    "Train ACC": train_eval["accuracy"],
                    "Valid ACC": valid_eval["accuracy"],
                    "Train F1": train_eval["f1"],
                    "Valid F1": valid_eval["f1"],
                    # "Train PRC": train_eval["precision"],
                    # "Valid PRC": valid_eval["precision"],
                    # "Train REC": train_eval["recall"],
                    # "Valid REC": valid_eval["recall"],
                    "Train Loss": train_loss,
                    "Valid Loss": valid_loss,
                }
            )
        else:
            wandb.log(
                {
                    "Train ACC": train_eval["accuracy"],
                    "Train F1": train_eval["f1"],
                    # "Train PRC": train_eval["precision"],
                    # "Train REC": train_eval["recall"],
                    "Train Loss": train_loss,
                }
            )

        # Checkpoint: (1) Better Accuracy (2) Better Loss if accuracy is the same as before
        if is_valid:
            if save_path and valid_eval["accuracy"] > best_acc:
                name = ckpt_name(model_type, pretrained_type, epoch, valid_eval["accuracy"], valid_loss, TIMESTAMP)
                best_acc = valid_eval["accuracy"]
                best_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(save_path, name))
                print(f'Model saved: {os.path.join(save_path, name)}')

            elif save_path and valid_eval["accuracy"] == best_acc and best_loss > valid_loss:
                name = ckpt_name(model_type, pretrained_type, epoch, valid_eval["accuracy"], valid_loss, TIMESTAMP)
                best_acc = valid_eval["accuracy"]
                best_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(save_path, name))
                print(f'Model saved: {os.path.join(save_path, name)}')
        else:
            if save_path and train_eval["accuracy"] > best_acc:
                name = ckpt_name(model_type, pretrained_type, epoch, train_eval["accuracy"], train_loss, TIMESTAMP)
                best_acc = train_eval["accuracy"]
                best_loss = train_loss
                torch.save(model.state_dict(), os.path.join(save_path, name))
                print(f'Model saved: {os.path.join(save_path, name)}')

            elif save_path and train_eval["accuracy"] == best_acc and best_loss > train_loss:
                name = f"{model_type}_{pretrained_type}_ep({epoch:0>2d})acc({train_eval['accuracy']:.4f})loss({train_loss})id({TIMESTAMP}).pth"
                best_loss = train_loss
                torch.save(model.state_dict(), os.path.join(save_path, name))
                print(f'Model saved: {os.path.join(save_path, name)}')



def validate(model, model_type, valid_loader, criterion):
    pred_list = []
    true_list = []
    total_loss = 0

    with torch.no_grad():
        model.eval()
        for sentences, labels in tqdm(valid_loader, desc="[Valid]"):
            if model_type == ModelType.SequenceClf:
                outputs = model(**sentences).logits
            elif model_type == ModelType.Base:
                outputs = model(**sentences).pooler_output
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
    TIMESTAMP = get_timestamp()
    LOAD_STATE_DICT = None
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default=ModelType.VanillaBert)
    parser.add_argument(
        "--pretrained-type", type=str, default=PreTrainedType.BertMultiLingual
    )
    parser.add_argument("--num-classes", type=int, default=Config.NumClasses)
    parser.add_argument("--poolder-idx", type=int, default=0)
    parser.add_argument("--load-state-dict", type=str, default=LOAD_STATE_DICT)
    parser.add_argument("--data-root", type=str, default=Config.Train)
    parser.add_argument("--tokenization-type", type=str, default=TokenizationType.Base)
    parser.add_argument("--epochs", type=int, default=Config.Epochs)
    parser.add_argument("--valid-size", type=int, default=Config.ValidSize)
    parser.add_argument("--train-batch-size", type=int, default=Config.Batch16)
    parser.add_argument("--valid-batch-size", type=int, default=512)
    parser.add_argument("--optim-type", type=str, default=Optimizer.Adam)
    parser.add_argument("--loss-type", type=str, default=Loss.CE)
    parser.add_argument("--lr", type=float, default=Config.LRSlow)
    parser.add_argument("--lr-scheduler", type=str, default=Optimizer.CosineScheduler)
    parser.add_argument("--device", type=str, default=Config.Device)
    parser.add_argument("--seed", type=int, default=Config.Seed)
    parser.add_argument("--save-path", type=str, default=Config.CheckPoint)

    # register logs to wandb
    args = parser.parse_args()
    name = args.model_type + "_" + args.pretrained_type + "_" + TIMESTAMP
    run = wandb.init(project="pstage-klue", name=name, reinit=True)
    wandb.config.update(args)

    # train
    VALID_CYCLE = int((TOTAL_SAMPLES * (1-args.valid_size)) / args.train_batch_size) - 1 # 학습 과정에서 2번만 검증
    print("=" * 100)
    print(args)
    print("=" * 100)
    train(**vars(args))

    # finish wandb session
    run.finish()
