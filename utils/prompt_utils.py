import logging

# Get an example
import torch
from torch.nn.functional import softmax
import os
import numpy as np
import tqdm
from tqdm import tqdm
from .metrics import evaluate_all
from sklearn.metrics import accuracy_score
from torch.nn.functional import softmax
from pathlib import Path


import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def eval_model(
    model,
    dataloader=None,
    yes_idx=None,
    no_idx=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    loss_crt=None,
):

    all_labels = []
    all_preds = []
    preds_acc = []

    dataset_loss = 0
    for idx_sample, (entry, label, mask_position) in enumerate(
        tqdm(dataloader, desc="eval")
    ):
        label_bin = 1 if (label.cpu().item() == 1) else 0
        entry_score_yes = []

        batch_iid = None
        batch_tti = None
        batch_im = None
        sample_loss = 0
        for idx_chk, (chunk, mask_pos) in enumerate(zip(entry, mask_position[0])):

            batch_iid = chunk["input_ids"].to(device)
            batch_tti = chunk["token_type_ids"].to(device)
            batch_im = chunk["attention_mask"].to(device)

            output = model(
                {
                    "input_ids": batch_iid,
                    "attention_mask": batch_im,
                    "token_type_ids": batch_tti,
                }
            )[0]

            batch_size = output.shape[0]
            logits = output[range(batch_size), mask_pos.item(), :]
            sample_loss += loss_crt(logits.to(device), label.to(device)).item()

            logits_yes = logits[:, yes_idx]
            logits_no = logits[:, no_idx]
            logits_yes_no = torch.stack((logits_no, logits_yes), dim=1)

            probs_yes_no = softmax(logits_yes_no, dim=1)
            entry_score_yes.append(torch.mean(probs_yes_no[:, 1]).cpu().item())

        sample_loss /= (idx_chk + 1)
        dataset_loss += sample_loss

        if len(entry_score_yes):
            pred_prob_yes = sum(entry_score_yes) / len(entry_score_yes)

            all_labels.append(label_bin)
            all_preds.append(pred_prob_yes)

    dataset_loss /= (idx_sample + 1)

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

    results["loss"] = dataset_loss
    results["acc"] = accuracy

    return results


def train_prompt_model(
    model,
    train_dataloader,
    train_dataloader_full,
    val_dataloader_full,
    optimizer,
    loss_crt,
    tb_writer,
    test_dataloader_full=None,
    epochs=200,
    scheduler=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    best_model_path="./checkpoints",
):
    Path(best_model_path).mkdir(parents=True, exist_ok=True)

    yes_idx = train_dataloader.dataset.yes_idx
    no_idx = train_dataloader.dataset.no_idx

    scheduler = scheduler(optimizer)

    best_overall_val = 0.0
    train_results_list = []
    val_results_list = []
    test_results_list = []

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
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }
            )
            logits = outputs[0][range(labels.shape[0]), mask_position.squeeze(), :]

            loss = loss_crt(logits, labels)
            loss.backward()

            epoch_train_loss += loss.cpu().item()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            train_results = eval_model(
                model,
                train_dataloader_full,
                yes_idx,
                no_idx,
                device,
                loss_crt=loss_crt,
            )
            val_results = eval_model(
                model,
                val_dataloader_full,
                yes_idx,
                no_idx,
                device,
                loss_crt=loss_crt,
            )

            total_epoch_train_loss = train_results["loss"]
            total_epoch_eval_loss = val_results["loss"]

            train_results_list.append(train_results)
            val_results_list.append(val_results)

            print("\t[Train] Results:", train_results)
            print("\t[Val] Results:", val_results)

            if val_results["overall"] > best_overall_val:
                best_overall_val = val_results["overall"]
                model.save_pretrained(best_model_path)

            if scheduler:
                scheduler.step(total_epoch_eval_loss)

            print("\tTrain loss: ", epoch_train_loss / (idx + 1))
            print("\tTotal train loss: ", total_epoch_train_loss)
            print("\tTotal eval loss: ", total_epoch_eval_loss)

            for metric in train_results:
                tb_writer.add_scalar(
                    f"train/{metric}", train_results[metric], epoch_idx + 1
                )
            for metric in val_results:
                tb_writer.add_scalar(
                    f"val/{metric}", val_results[metric], epoch_idx + 1
                )

            if test_dataloader_full:
                test_results = eval_model(
                    model,
                    test_dataloader_full,
                    yes_idx,
                    no_idx,
                    device,
                    loss_crt=loss_crt,
                )
                total_epoch_train_loss = test_results["loss"]
                test_results_list.append(test_results)
                print("\tTotal train loss: ", total_epoch_train_loss)

                for metric in test_results:
                    tb_writer.add_scalar(
                        f"test/{metric}", test_results[metric], epoch_idx + 1
                    )

    if test_dataloader_full:
        return {
            "train": train_results_list,
            "val": val_results_list,
            "test": test_results_list,
        }
    else:
        return {"train": train_results_list, "val": val_results_list}
