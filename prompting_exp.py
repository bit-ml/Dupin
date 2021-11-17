import torch
from torch.nn.functional import softmax
import os
import numpy as np
import tqdm
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTJForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)
from transformers import BertForMaskedLM, AdamW, PreTrainedTokenizerBase
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.nn.functional import softmax
import random

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainablePromptModel:
    def __init__(
        self, model_name="bert-base-uncased", model_type=AutoModelForMaskedLM, **kwargs
    ):
        self.model = model_type.from_pretrained(model_name, **kwargs)
        self.tokenizer = model_type.from_pretrained(model_name, **kwargs)

    def __call__(self, **kwargs):
        return self.model(**kwargs)


from utils import train_prompt_model
from datasets import RedditPromptDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


model = TrainablePromptModel()
trainable_params = [p for (n, p) in model.model.named_parameters() if "bias" in n]

train_dataset_path = "/darkweb_ds/reddit_darknet/reddit_open_split/train"
val_dataset_path = "/darkweb_ds/reddit_darknet/reddit_open_split/val"
test_dataset_path = "/darkweb_ds/reddit_darknet/reddit_open_split/test"

train_dataset = RedditPromptDataset(
    path=train_dataset_path,
    tokenizer=model.tokenizer,
    debug=False,
    train=True,
)
train_dataset_full = RedditPromptDataset(
    path=train_dataset_path,
    tokenizer=model.tokenizer,
    debug=False,
    train=False,
)
val_dataset_full = RedditPromptDataset(
    path=val_dataset_path, tokenizer=model.tokenizer, debug=False, train=False
)
test_dataset_full = RedditPromptDataset(
    path=test_dataset_path,
    tokenizer=model.tokenizer,
    debug=False,
    train=False,
)

train_sampler = RandomSampler(train_dataset)
train_sampler_full = SequentialSampler(train_dataset_full)
val_sampler_full = SequentialSampler(val_dataset_full)
test_sampler_full = SequentialSampler(test_dataset_full)

BATCH_SIZE = 16

train_dataloader = DataLoader(
    train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE
)

train_dataloader_full = DataLoader(
    train_dataset_full, sampler=train_sampler_full, batch_size=1
)

val_dataloader_full = DataLoader(
    val_dataset_full, sampler=val_sampler_full, batch_size=1
)

test_dataloader_full = DataLoader(
    test_dataset_full, sampler=test_sampler_full, batch_size=1
)

train_prompt_model(
    {"model": model, "trainable_params": trainable_params},
    train_dataloader=train_dataloader,
    train_dataloader_full=train_dataloader_full,
    val_dataloader_full=val_dataloader_full,
    test_dataloader_full=test_dataloader_full,
)
