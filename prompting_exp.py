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
    AutoModelForCausalLM
)
from transformers import BertForMaskedLM, AdamW, PreTrainedTokenizerBase
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.nn.functional import softmax
import random

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TrainablePromptModel:
    def __init__(self, model_name='bert-base-uncased', model_type=AutoModelForMaskedLM, **kwargs):
        self.model = model_type.from_pretrained(model_name, **kwargs)
        self.tokenizer = model_type.from_pretrained(model_name, **kwargs)


x = TrainablePromptModel()

print(x.model)
