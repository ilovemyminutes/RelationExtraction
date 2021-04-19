import argparse
import os
import warnings
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from evaluation import evaluate
from models import load_model
from dataset import REDataset, split_train_test_loader
from optimizers import get_optimizer, get_scheduler
from criterions import get_criterion
from utils import get_timestamp, get_timestamp, set_seed, verbose, ckpt_name, save_json
from config import ModelType, Config, Optimizer, PreTrainedType, PreProcessType, Loss

warnings.filterwarnings("ignore")


TOTAL_SAMPLES = 9000


def train(
    model_type: str = ModelType.VanillaBert,  # 불러올 모델 프레임
    pretrained_type: str = PreTrainedType.MultiLingual,  # 모델에 활용할 Pretrained BERT Backbone 이름
    num_classes: int = Config.NumClasses,  # 카테고리 수
    pooler_idx: int = 0,  # 인코딩 결과로부터 추출할 hidden state. 0: [CLS]
    dropout: float = 0.8,
    load_state_dict: str = None,  # (optional) 저장한 weight 경로
    data_root: str = Config.Train,  # 학습 데이터 경로
    preprocess_type: str = PreProcessType.Base,  # 텍스트 전처리 타입
    epochs: int = Config.Epochs,
    valid_size: float = Config.ValidSize,  # 학습 데이터 중 검증에 활용할 데이터 비율
    train_batch_size: int = Config.Batch32,
    valid_batch_size: int = 512,
    optim_type: str = Optimizer.Adam,
    loss_type: str = Loss.CE,
    lr: float = Config.LR,
    lr_scheduler: str = Optimizer.CosineAnnealing,
    device: str = Config.Device,
    seed: int = Config.Seed,
    save_path: str = Config.CheckPoint,
):
    # set seed
    set_seed(seed)

    # load data
    dataset = REDataset(root=data_root, preprocess_type=preprocess_type, device=device)

    # 학습 데이터를 분리하지 않을 경우
    if valid_size == 0:
        is_valid = False  # validation flag
        train_loader = DataLoader(
            dataset, batch_size=train_batch_size, shuffle=True, drop_last=True
        )

    # 학습 데이터를 학습/검증 데이터로 분리할 경우
    else:
        is_valid = True  # validation flag
        train_loader, valid_loader = split_train_test_loader(
            dataset=dataset,
            test_size=valid_size,
            train_batch_size=train_batch_size,
            test_batch_size=valid_batch_size,
        )

    # load model
    model = load_model(
        model_type, pretrained_type, num_classes, load_state_dict, pooler_idx, dropout
    )
    model.to(device)
    model.train()

    # load criterion, optimizer, scheduler
    criterion = get_criterion(type=loss_type)
    optimizer = get_optimizer(model=model, type=optim_type, lr=lr)
    if lr_scheduler is not None:
        scheduler = get_scheduler(
            type=lr_scheduler, optimizer=optimizer, num_training_steps=TOTAL_STEPS
        )

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

        for sentences, labels in tqdm(train_loader, desc="[Train]"):
            if model_type == ModelType.SequenceClf:
                loss, outputs = model(**sentences, labels=labels).values()
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

            # record predicted outputs
            pred_list.append(preds)
            true_list.append(labels)

            if epoch == 0:
                pred_arr = np.hstack(pred_list)
                true_arr = np.hstack(true_list)

                # evaluate each step
                train_eval = evaluate(
                    y_true=true_arr, y_pred=pred_arr
                )  # ACC, F1, PRC, REC
                train_loss = total_loss / len(true_arr)

                # save logs of train step
                wandb.log(
                    {
                        f"First EP Train ACC": train_eval["accuracy"],
                        f"First EP Train F1": train_eval["f1"],
                        f"First EP Train Loss": train_loss,
                    }
                )

        # evaluate each epoch
        pred_arr = np.hstack(pred_list)
        true_arr = np.hstack(true_list)

        train_eval = evaluate(y_true=true_arr, y_pred=pred_arr)  # ACC, F1, PRC, REC
        train_loss = total_loss / len(true_arr)

        # validation phase
        valid_eval, valid_loss = validate(
            model=model,
            model_type=model_type,
            valid_loader=valid_loader,
            criterion=criterion,
        )
        verbose(phase="Valid", eval=valid_eval, loss=valid_loss)
        verbose(phase="Train", eval=train_eval, loss=train_loss)

        # logs for each epoch of train, valid both
        if is_valid:  # when train set splited to train/valid
            wandb.log(
                {
                    "Train ACC": train_eval["accuracy"],
                    "Valid ACC": valid_eval["accuracy"],
                    "Train F1": train_eval["f1"],
                    "Valid F1": valid_eval["f1"],
                    "Train Loss": train_loss,
                    "Valid Loss": valid_loss,
                }
            )
        else:  # when train set not splited - all train set are feeded to train
            wandb.log(
                {
                    "Train ACC": train_eval["accuracy"],
                    "Train F1": train_eval["f1"],
                    "Train Loss": train_loss,
                }
            )

        # Checkpoint: (1) Better Accuracy (2) Better Loss if accuracy is the same as before
        if is_valid:
            if save_path and valid_eval["accuracy"] > best_acc:
                name = ckpt_name(
                    model_type,
                    pretrained_type,
                    epoch,
                    valid_eval["accuracy"],
                    valid_loss,
                    TIMESTAMP,
                )
                best_acc = valid_eval["accuracy"]
                best_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(save_path, name))
                print(f"Model saved: {os.path.join(save_path, name)}")

            elif (
                save_path
                and valid_eval["accuracy"] == best_acc
                and best_loss > valid_loss
            ):
                name = ckpt_name(
                    model_type,
                    pretrained_type,
                    epoch,
                    valid_eval["accuracy"],
                    valid_loss,
                    TIMESTAMP,
                )
                best_acc = valid_eval["accuracy"]
                best_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(save_path, name))
                print(f"Model saved: {os.path.join(save_path, name)}")
        else:
            if save_path and train_eval["accuracy"] > best_acc:
                name = ckpt_name(
                    model_type,
                    pretrained_type,
                    epoch,
                    train_eval["accuracy"],
                    train_loss,
                    TIMESTAMP,
                )
                best_acc = train_eval["accuracy"]
                best_loss = train_loss
                torch.save(model.state_dict(), os.path.join(save_path, name))
                print(f"Model saved: {os.path.join(save_path, name)}")

            elif (
                save_path
                and train_eval["accuracy"] == best_acc
                and best_loss > train_loss
            ):
                name = f"{model_type}_{pretrained_type}_ep({epoch:0>2d})acc({train_eval['accuracy']:.4f})loss({train_loss})id({TIMESTAMP}).pth"
                best_loss = train_loss
                torch.save(model.state_dict(), os.path.join(save_path, name))
                print(f"Model saved: {os.path.join(save_path, name)}")


def validate(model, model_type, valid_loader, criterion):
    pred_list = []
    true_list = []
    total_loss = 0
    model.eval()

    with torch.no_grad():
        for sentences, labels in tqdm(valid_loader, desc="[Valid]"):
            if model_type == ModelType.SequenceClf:
                loss, outputs = model(**sentences, labels=labels).values()
            else:
                outputs = model(**sentences)
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
    TIMESTAMP = get_timestamp()  # used as an identifier in model save phase
    LOAD_STATE_DICT = "./saved_models/VanillaBert_v2_bert-base-multilingual-cased_20210419131744/VanillaBert_v2_bert-base-multilingual-cased_ep(17)acc(0.7189)loss(0.0042)id(20210419131744).pth"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default=ModelType.VanillaBert_v2)
    parser.add_argument(
        "--pretrained-type", type=str, default=PreTrainedType.MultiLingual
    )
    parser.add_argument("--num-classes", type=int, default=Config.NumClasses)
    parser.add_argument("--pooler-idx", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.8)
    parser.add_argument("--load-state-dict", type=str, default=LOAD_STATE_DICT)
    parser.add_argument("--data-root", type=str, default=Config.Train)
    parser.add_argument("--preprocess-type", type=str, default=PreProcessType.ES)
    parser.add_argument("--epochs", type=int, default=Config.Epochs)
    parser.add_argument("--valid-size", type=int, default=Config.ValidSize)
    parser.add_argument("--train-batch-size", type=int, default=Config.Batch8)
    parser.add_argument("--valid-batch-size", type=int, default=512)
    parser.add_argument("--optim-type", type=str, default=Optimizer.Adam)
    parser.add_argument("--loss-type", type=str, default=Loss.LS)
    parser.add_argument("--lr", type=float, default=Config.LRSlow)
    parser.add_argument("--lr-scheduler", type=str, default=Optimizer.CosineAnnealing)
    parser.add_argument("--device", type=str, default=Config.Device)
    parser.add_argument("--seed", type=int, default=Config.Seed)
    parser.add_argument("--save-path", type=str, default=Config.CheckPoint)

    args = parser.parse_args()
    if args.model_type == ModelType.SequenceClf:
        args.loss_type = Loss.CE

    # register logs to wandb
    name = (
        args.model_type + "_" + args.pretrained_type + "_" + TIMESTAMP
    )  # save file name: [MODEL-TYPE]_[PRETRAINED-TYPE]_[EPOCH][ACC][LOSS][ID].pth
    run = wandb.init(project="pstage-klue", name=name, reinit=True)
    wandb.config.update(args)

    # make checkpoint directory to save model during train
    checkpoint_dir = f"{args.model_type}_{args.pretrained_type}_{TIMESTAMP}"
    if checkpoint_dir not in os.listdir(args.save_path):
        os.mkdir(os.path.join(args.save_path, checkpoint_dir))
    args.save_path = os.path.join(args.save_path, checkpoint_dir)

    # save param dict
    save_param = vars(args)
    save_param['device'] = save_param['device'].type
    save_json(os.path.join(args.save_path, 'param_dict.json'), save_param)

    # train
    TOTAL_STEPS = args.epochs * (
        int(TOTAL_SAMPLES * (1 - args.valid_size)) // args.train_batch_size
    )
    print("=" * 100)
    print(args)
    print("=" * 100)
    train(**vars(args))

    # finish wandb session
    run.finish()
