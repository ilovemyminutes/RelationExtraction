import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import REDataset
from config import Config, ModelType, TokenizationType
from models import load_model


def predict(
    load_state_dict,
    num_classes,
    pooler_idx,
    data_root,
    tokenization_type,
    device,
    save_path,
):
    # load model
    model_type, pretrained_type = get_model_pretrained_type(load_state_dict)
    model = load_model(
        model_type, pretrained_type, num_classes, load_state_dict, pooler_idx
    )
    model.to(device)
    model.eval()

    # load dataset
    dataset = REDataset(data_root, tokenization_type, device)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False)

    # inference phase
    pred_list = []
    with torch.no_grad():
        for sentences, _ in tqdm(dataloader, desc="[Inference]"):
            if model_type == ModelType.SequenceClf:
                outputs = model(**sentences).logits
            elif model_type == ModelType.Base:
                outputs = model(**sentences).pooler_output
            else:
                outputs = model(**sentences)

            _, preds = torch.max(outputs, dim=1)
            preds = preds.data.cpu().numpy()
            pred_list.append(preds)
    pred_arr = np.hstack(pred_list)

    # export phase
    submission = pd.DataFrame(dict(pred=pred_arr.tolist()))

    if save_path:
        model_name = os.path.basename(load_state_dict)
        if model_name not in os.listdir(save_path):
            os.mkdir(os.path.join(save_path, model_name))
        save_path = os.path.join(save_path, model_name)
        fname = f"submission_{model_name}.csv"
        submission.to_csv(os.path.join(save_path, fname), index=False)
    else:
        return submission


def get_model_pretrained_type(load_state_dict: str):
    basename = os.path.basename(load_state_dict)
    model_type = basename.split("_")[0]
    pretrained_type = basename.split("_")[1]
    return model_type, pretrained_type


if __name__ == "__main__":
    MODELNAME = "VanillaBert_bert-base-multilingual-cased_20210417181839/VanillaBert_bert-base-multilingual-cased_ep(02)acc(0.4878)loss(0.0048)id(20210417181839).pth"
    LOAD_STATE_DICT = os.path.join(Config.CheckPoint, MODELNAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--load-state-dict", type=str, default=LOAD_STATE_DICT)
    parser.add_argument("--num-classes", type=int, default=Config.NumClasses)
    parser.add_argument("--pooler-idx", type=int, default=0)
    parser.add_argument("--data-root", type=str, default=Config.Test)
    parser.add_argument("--tokenization-type", type=str, default=TokenizationType.Base)
    parser.add_argument("--device", type=str, default=Config.Device)
    parser.add_argument("--save-path", type=str, default=Config.SavePath)
    args = parser.parse_args()

    predict(**vars(args))