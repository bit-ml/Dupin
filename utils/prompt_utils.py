import logging

# Get an example
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
)
from transformers import BertForMaskedLM, AdamW, PreTrainedTokenizerBase
from promptsource.templates import TemplateCollection
from datasets import load_dataset
from metrics import evaluate_all
from sklearn.metrics import accuracy_score
from torch.nn.functional import softmax
import random

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def eval_model(
    model,
    dataloader=None,
    tokenizer=None,
    yes_idx=None,
    no_idx=None,
    device=None,
    loss_crt=None
):
# todo: clean and add +ret loss crt
    all_labels = []
    all_preds = []
    preds_acc = []

    for entry, label, mask_position in tqdm(dataloader, desc="eval"):
        label = label
        entry_pred = []
        entry_score = []
        entry_score_yes = []
        entry_score_no = []

        batch_iid = None
        batch_tti = None
        batch_im = None
        for idx, (chunk, mask_pos) in enumerate(zip(entry, mask_position[0])):

            batch_iid = chunk["input_ids"].cuda()
            batch_tti = chunk["token_type_ids"].cuda()
            batch_im = chunk["attention_mask"].cuda()

            output = model(
                batch_iid,
                token_type_ids=batch_tti,
                attention_mask=batch_im
            )[0]

            batch_size = output.shape[0]
            logits = output[range(batch_size), mask_pos.item(), :]

            logits_yes = logits[:, yes_idx]
            logits_no = logits[:, no_idx]
            logits_yes_no = torch.stack((logits_no, logits_yes), dim=1)

            probs_yes_no = softmax(logits_yes_no, dim=1)
            entry_score_yes.append(torch.mean(probs_yes_no[:, 1]).cpu().item())

        if len(entry_score_yes):
            pred_prob_yes = sum(entry_score_yes) / len(entry_score_yes)
            label_cpu = label.cpu().item()

            all_labels.append(label_cpu)
            all_preds.append(pred_prob_yes)

    gt = np.array(all_labels)
    pred = np.array(all_preds)

    # sanity check:
    assert len(gt) == len(pred)

    gt = np.array(gt, dtype=np.float64)
    pred = np.array(pred, dtype=np.float64)

    preds_acc = [0 if p < 0.5 else 1 for p in pred]

    assert len(gt) == len(pred)

    accuracy = accuracy_score(gt, preds_acc)

    results = evaluate_all(gt, pred)
    results["acc"] = accuracy

    return results


def train_prompt_model(
    model_dict,
    train_dataloader,
    val_dataloader,
    epochs=200,
    wd=1e-4,
    lr=1e-4,
    scheduler=None,
    test_dataloader=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):

    yes_idx = train_dataloader.dataset.yes_idx
    no_idx = val_dataloader.dataset.no_idx

    optimizer = AdamW(model_dict["model_params"], lr=lr, weight_decay=wd)
    model = model_dict["model"]

    loss_crt = CrossEntropyLoss()

    best_overall_val = 0.0
    train_results_list = []
    val_results_list = []
    for epoch_idx in range(epochs):
        print("epoch ", epoch_idx)

        loss = 100
        epoch_train_loss = 0.0
        for idx, (batch, labels, mask_position) in enumerate(
            tqdm(train_dataloader, desc=f"[Train] Loss: {loss}")
        ):
            labels = labels.to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs[0][range(labels.shape[0]), mask_position.squeeze(), :]

            loss = loss_crt(logits, labels)
            loss.backward()

            epoch_train_loss += loss.cpu().item()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            train_results = eval_model(
                model, train_dataloader_eval, tokenizer, yes_idx, no_idx, device, loss_crt=loss_crt
            )
            val_results = eval_model(
                model, val_dataloader, tokenizer, yes_idx, no_idx, device, loss_crt=loss_crt
            )

            total_epoch_train_loss = train_results["loss"]
            total_epoch_eval_loss = val_results["loss"]

            train_results_list.append(train_results)
            val_results_list.append(val_results)

            print("\t[Train] Results:", train_results)
            print("\t[Val] Results:", val_results)

            if val_results["overall"] > best_overall_val:
                best_overall_val = val_results["overall"]
                best_val_results = val_results
                best_val_epoch = epoch_idx
                model.save_pretrained(best_model_path)

            scheduler.step(total_epoch_eval_loss)

        print("\tTrain loss: ", epoch_train_loss / (idx + 1))
        print("\tEval loss: ", total_epoch_eval_loss)
