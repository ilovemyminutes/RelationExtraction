import argparse
import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig


def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
    model.eval()
    output_pred = []
  
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
                )
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)

            output_pred.append(result)
  
    return np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label



class RE_Dataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label})
    return out_dataset

def load_data(dataset_dir):
    with open('./input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    dataset = preprocessing_dataset(dataset, label_type)
    return dataset

def tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
        )
    return tokenized_sentences


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def train():
    MODEL_NAME = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # load dataset
    train_dataset = load_data("./input/data/train/train.tsv")

    #dev_dataset = load_data("./dataset/train/dev.tsv")
    train_label = train_dataset['label'].values

    tokenized_train = tokenized_dataset(train_dataset, tokenizer)

    RE_train_dataset = RE_Dataset(tokenized_train, train_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    bert_config = BertConfig.from_pretrained(MODEL_NAME)
    bert_config.num_labels = 42
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config)
    model.parameters
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./result",
        save_total_limit=3,
        save_steps=500,
        num_train_epochs=15,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RE_train_dataset,
    )

    trainer.train()


def main(args):
    """
        주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    # train()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    TOK_NAME = "bert-base-multilingual-cased"  
    tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

    # load my model
    MODEL_NAME = args.model_dir # model dir.
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.parameters
    model.to(device)

    # load test datset
    test_dataset_dir = "./input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset ,test_label)

    # predict answer
    pred_answer = inference(model, test_dataset, device)
    # make csv file with predicted answer
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv('./predictions/submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="./result/checkpoint-8000")
    args = parser.parse_args()
    print(args)
    main(args)