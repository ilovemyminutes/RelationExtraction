import os
import json
import pickle
import random
import time
import warnings
from filelock import FileLock
from typing import Dict, List, Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from transformers.utils import logging
logger = logging.get_logger(__name__)
from config import Config


class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        short_seq_probability=0.1, # 지정한 길이보다 시퀀스를 짧게 만들 확률(길이는 2~MAX_LEN의 랜덤으로 지정됨)
        nsp_probability=0.5,
    ):
        # 여기 부분은 학습 데이터를 caching하는 부분입니다 :-)
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_nsp_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        self.tokenizer = tokenizer

        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")
                # 여기서부터 본격적으로 dataset을 만듭니다.
                self.documents = [[]] # 리스트 내 2차원 리스트를 갖는데, 바깥 차원은 하나의 문서를, 안쪽 차원은 하나의 문장을 의미
                with open(file_path, encoding="utf-8") as f:
                    while True: # 일단 문장을 읽고
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()

                        # 이중 띄어쓰기가 발견된다면, 나왔던 문장들을 모아 하나의 문서로 묶어버립니다.
                        # 즉, 문단 단위로 데이터를 저장합니다.
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        
                        # 토큰화
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)  # 마지막 문장 성분에 토큰들을 추가

                # 이제 코퍼스 전체를 읽고, 문서 데이터를 생성했습니다! :-)
                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                # 본격적으로 학습을 위한 데이터로 변형시켜볼까요?
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index) # 함수로 가봅시다.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int):
        """Creates examples for a single document."""
        # 문장의 앞, 뒤에 [CLS], [SEP] token이 부착되기 때문에, 내가 지정한 size에서 2 만큼 빼줍니다.
            # 실제 입력될 토큰 수는 설정한 최대 토큰에서 스페셜 토큰 수를 뺀 만큼
        # 예를 들어 128 token 만큼만 학습 가능한 model을 선언했다면, 학습 데이터로부터는 최대 126 token만 가져오게 됩니다.
        max_num_tokens = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=True)

        # We *usually* want to fill up the entire sequence since we are padding
        # to `block_size` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pretraining and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `block_size` is a hard limit.

        # 여기가 재밌는 부분인데요!
        # 위에서 설명했듯이, 학습 데이터는 126 token(128-2)을 채워서 만들어지는게 목적입니다.
        # 하지만 나중에 BERT를 사용할 때, 126 token 이내의 짧은 문장을 테스트하는 경우도 분명 많을 것입니다 :-)
        # 그래서 short_seq_probability 만큼의 데이터에서는 2-126 사이의 random 값으로 학습 데이터를 만들게 됩니다.
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_probability:
            target_seq_length = random.randint(2, max_num_tokens)

        current_chunk = []  # a buffer stored current working segments
        current_length = 0
        i = 0

        # 데이터 구축의 단위는 document 입니다
        # 이 때, 무조건 문장_1[SEP]문장_2 이렇게 만들어지는 것이 아니라,
        # 126 token을 꽉 채울 수 있게 문장_1+문장_2[SEP]문장_3+문장_4 형태로 만들어질 수 있습니다.
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1 # 문장 A에 포함할 raw 문장 수
                    # 여기서 문장_1+문장_2 가 이루어졌을 때, 길이를 random하게 짤라버립니다 :-)
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    tokens_a = []
                    for j in range(a_end): # 첫 문장의 길이가 일정 범위 내에서 랜덤하게 expand
                        tokens_a.extend(current_chunk[j]) # 문장 단위로 포함함


                    # 문장 B 생성
                    # nsp_probabilty 확률에 따라, 인접한 문장으로 구성할지 그렇지 않은 문장을 구성할지 결정
                    tokens_b = []
                    # 50%의 확률로 랜덤하게 다른 문장을 선택하거나, 다음 문장을 학습데이터로 만듭니다.
                    if len(current_chunk) == 1 or random.random() < self.nsp_probability:
                        is_random_next = True # "다음 문장은 랜덤(인접 문장X)"
                        target_b_length = target_seq_length - len(tokens_a) # B는 A가 차지한 부분을 뺀 나머지 길이에서 놀아야겠지

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10): # 효율을 위해 10으로 설정했을까?
                            random_document_index = random.randint(0, len(self.documents) - 1)
                            if random_document_index != doc_index: # 다른 문서로부터 문장 추출
                                break
                        # 여기서 랜덤하게 선택합니다 :-)
                        random_document = self.documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1) # 임의 추출할 최초 지점 임의 설정
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j]) # 남은 길이 채워버리기
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False # "다음 문장은 인접한 문장이야"
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    # 이제 126 token을 넘는다면 truncation을 해야합니다.
                    # 이 때, 126 token 이내로 들어온다면 행위를 멈추고,
                    # 만약 126 token을 넘는다면, segmentA와 segmentB에서 랜덤하게 하나씩 제거합니다.
                    def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                        """Truncates a pair of sequences to a maximum sequence length."""
                        while True:
                            total_length = len(tokens_a) + len(tokens_b)
                            if total_length <= max_num_tokens:
                                break

                            # 길이가 더 긴 문장의 토큰 일부를 제거하는 식으로 truncation
                            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                            assert len(trunc_tokens) >= 1
                            # We want to sometimes truncate from the front and sometimes from the
                            # back to add more randomness and avoid biases.

                            # 랜덤하게 앞에서 빼거나 뒤에서 빼거나
                            if random.random() < 0.5: 
                                del trunc_tokens[0]
                            else:
                                trunc_tokens.pop()

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    # 총 길이가 각각 최소 1 이상인지 확인
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    # add special tokens
                    input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    # add token type ids, 0 for sentence a, 1 for sentence b
                    token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
                    
                    # 드디어 아래 항목에 대한 데이터셋이 만들어졌습니다! :-)
                    # 즉, segmentA[SEP]segmentB, [0, 0, .., 0, 1, 1, ..., 1], NSP 데이터가 만들어진 것입니다 :-)
                    # 그럼 다음은.. 이 데이터에 [MASK] 를 씌워야겠죠?
                    example = {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                        "next_sentence_label": torch.tensor(1 if is_random_next else 0, dtype=torch.long),
                    }

                    self.examples.append(example)

                current_chunk = []
                current_length = 0

            i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# Dataset 구성.
class RE_Dataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == "blind":
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame(
        {
            "sentence": dataset[1],
            "entity_01": dataset[2],
            "entity_02": dataset[5],
            "label": label,
        }
    )
    return out_dataset


# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
    # load label_type, classes
    label_path = Config.LabelType if os.path.isfile(Config.LabelType) else "../input/data/label_type.pkl"
    with open(label_path, "rb") as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter="\t", header=None)
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset


# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset["entity_01"], dataset["entity_02"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
        add_special_tokens=True,
    )
    return tokenized_sentences


if __name__ == '__main__':
    tokenizer = BertTokenizer(
    vocab_file="./samples/my_tokenizer-vocab.txt",
    max_len=128,
    do_lower_case=False
    )

    dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path='./samples/wiki_20190620_small.txt',
    block_size=128,
    overwrite_cache=False,
    short_seq_probability=0.1,
    nsp_probability=0.5,
    )

data_collator = DataCollatorForLanguageModeling(    # [MASK] 를 씌우는 것은 저희가 구현하지 않아도 됩니다! :-)
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)