import logging
import torch
import os
import numpy as np
import tqdm
import warnings
from tqdm import tqdm
from .metrics import evaluate_all
from sklearn.metrics import accuracy_score
from torch.nn.functional import softmax
from pathlib import Path
from models import TrainablePromptModel, TrainableClfModel

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def get_logits_prompt_train(output, labels, mask_position):
    return output[0][range(labels.shape[0]), mask_position.squeeze(), :]


def get_logits_prompt_eval(output, batch_size, mask_pos):
    return output[range(batch_size), mask_pos.squeeze(), :]


def get_logits_clf_train(output):
    logits = output.logits
    return logits


def get_logits_clf_eval(output):
    return output


def train_model(
    model,
    train_dataloader,
    train_dataloader_full,
    val_dataloader_full,
    tb_writer,
    optimizer,
    loss_crt,
    get_logits_train_fun=None,
    get_logits_eval_fun=None,
    test_dataloader_full=None,
    epochs=200,
    scheduler=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    best_model_path="./checkpoints",
    infer_functions=True,
):
    if infer_functions and get_logits_train_fun is None:
        if isinstance(model, TrainablePromptModel):
            get_logits_train_fun = get_logits_prompt_train
        elif isinstance(model, TrainableClfModel):
            get_logits_train_fun = get_logits_clf_train
        else:
            raise Exception(
                f"Model {type(model)} not supported, please pass a valid function to get_logits_train."
            )

    if infer_functions and get_logits_eval_fun is None:
        if isinstance(model, TrainablePromptModel):
            get_logits_eval_fun = get_logits_prompt_eval
        elif isinstance(model, TrainableClfModel):
            get_logits_eval_fun = get_logits_clf_eval
        else:
            raise Exception(
                f"Model {type(model)} not supported, please pass a valid function to get_logits_eval."
            )

    Path(best_model_path).mkdir(parents=True, exist_ok=True)

    scheduler = scheduler(optimizer, verbose=True)

    best_overall_val = 0.0
    best_epoch_loss = 999
    train_results_list = []
    val_results_list = []
    test_results_list = []

    yes_idx = train_dataloader.dataset.yes_idx
    no_idx = train_dataloader.dataset.no_idx

    for epoch_idx in range(epochs):
        print("Epoch ", epoch_idx)

        loss = 100
        epoch_train_loss = 0.0
        for idx, ds_sample in enumerate(
            tqdm(train_dataloader, desc=f"[Train] Loss: {loss}")
        ):
            if isinstance(model, TrainablePromptModel):
                (batch, labels, mask_position) = ds_sample
            elif isinstance(model, TrainableClfModel):
                (batch, labels) = ds_sample
            else:
                warnings.warn("Unknown model, classification fallback.")
                (batch, labels) = ds_sample

            labels = labels.to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            output = model(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }
            )

            if isinstance(model, TrainablePromptModel):
                logits = get_logits_train_fun(output, labels, mask_position)
            elif isinstance(model, TrainableClfModel):
                logits = get_logits_train_fun(output)
                # Dataset returns labels with the 'yes' token id and 'no' token id
                labels[labels == yes_idx] = 1
                labels[labels == no_idx] = 0
            else:
                warnings.warn("Unknown model, classification fallback.")
                logits = get_logits_train_fun(output)

            loss = loss_crt(logits, labels)
            loss.backward()

            epoch_train_loss += loss.cpu().item()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            # train_results = eval_model(
            #     model,
            #     train_dataloader_full,
            #     yes_idx,
            #     no_idx,
            #     device,
            #     get_logits_fun=get_logits_eval_fun,
            #     loss_crt=loss_crt,
            # )
            val_results = eval_model_index(
                model,
                val_dataloader_full,
                yes_idx,
                no_idx,
                device,
                get_logits_fun=get_logits_eval_fun,
                loss_crt=loss_crt,
            )

            # total_epoch_train_loss = train_results["loss"]
            total_epoch_eval_loss = val_results["loss"]

            # train_results_list.append(train_results)
            val_results_list.append(val_results)

            # print("\t[Train] Results:", train_results)
            print("\t[Val] Results:", val_results)

            if val_results["overall"] > best_overall_val:
                best_overall_val = val_results["overall"]
                save_path = os.path.join(best_model_path, "/overall")
                model.save_pretrained(save_path)
                print(f'Overall better, saving model to {save_path}...')

            if total_epoch_eval_loss < best_epoch_loss:
                best_epoch_loss = total_epoch_eval_loss
                save_path = os.path.join(best_model_path, "/loss")
                model.save_pretrained(save_path)
                print(f'Loss better, saving model to {save_path}...')

            if scheduler:
                scheduler.step(total_epoch_eval_loss)

            # print("\tTrain loss: ", epoch_train_loss / (idx + 1))
            # print("\tTotal train loss: ", total_epoch_train_loss)
            print("\tTotal eval loss: ", total_epoch_eval_loss)

            # for metric in train_results:
            #     tb_writer.add_scalar(
            #         f"train/{metric}", train_results[metric], epoch_idx + 1
            #     )
            for metric in val_results:
                tb_writer.add_scalar(
                    f"val/{metric}", val_results[metric], epoch_idx + 1
                )

            if test_dataloader_full:
                test_results = eval_model_index(
                    model,
                    test_dataloader_full,
                    yes_idx,
                    no_idx,
                    device,
                    get_logits_fun=get_logits_eval_fun,
                    loss_crt=loss_crt,
                )
                total_epoch_test_loss = test_results["loss"]
                test_results_list.append(test_results)
                print("\t[Test] Results:", test_results)
                print("\tTotal test loss: ", total_epoch_test_loss)

                for metric in test_results:
                    tb_writer.add_scalar(
                        f"test/{metric}", test_results[metric], epoch_idx + 1
                    )

    if test_dataloader_full:
        return {
            # "train": train_results_list,
            "val": val_results_list,
            "test": test_results_list,
        }
    else:
        return {"train": train_results_list, "val": val_results_list}


def eval_model(
    model,
    dataloader=None,
    yes_idx=None,
    no_idx=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    loss_crt=None,
    get_logits_fun=None,
    max_batch_size=1,
):

    all_labels = []
    all_preds = []
    preds_acc = []

    dataset_loss = 0
    idx_sample = 0
    for idx_sample, ds_sample in enumerate(
        tqdm(dataloader, desc="eval")
    ):
        if isinstance(model, TrainablePromptModel):
            (entry, label, mask_position) = ds_sample
            mask_position = mask_position.squeeze(0)
        elif isinstance(model, TrainableClfModel):
            (entry, label) = ds_sample
        else:
            warnings.warn("Unknown model, classification fallback.")
            (entry, label) = ds_sample

        label_bin = 1 if (label.cpu().item() == yes_idx) else 0
        entry_score_yes = []

        sample_loss = 0
        idx_split = 0
        while entry:
            batch = entry[:max_batch_size]
            batch_iid = torch.stack([e["input_ids"].squeeze() for e in batch]).to(
                device
            )
            batch_tti = torch.stack([e["token_type_ids"].squeeze() for e in batch]).to(
                device
            )
            batch_am = torch.stack([e["attention_mask"].squeeze() for e in batch]).to(
                device
            )

            entry = entry[max_batch_size:]
            if isinstance(model, TrainablePromptModel):
                batch_mask_pos = mask_position[:max_batch_size]
                mask_position = mask_position[max_batch_size:]

            output = model(
                {
                    "input_ids": batch_iid,
                    "attention_mask": batch_am,
                    "token_type_ids": batch_tti,
                }
            )[0]

            batch_size = output.shape[0]

            if isinstance(model, TrainablePromptModel):
                logits = get_logits_fun(output, batch_size, batch_mask_pos)
                label_batch = label.repeat(batch_size)
                logits_yes = logits[:, yes_idx]
                logits_no = logits[:, no_idx]

            elif isinstance(model, TrainableClfModel):
                logits = get_logits_fun(output)
                label_batch = torch.tensor(label_bin).repeat(batch_size)
                logits_yes = logits[:, 1]
                logits_no = logits[:, 0]
                
            logits_yes_no = torch.stack((logits_no, logits_yes), dim=1)

            probs_yes_no = softmax(logits_yes_no, dim=1)
            entry_score_yes.append(torch.mean(probs_yes_no[:, 1]).cpu().item())
            sample_loss += loss_crt(logits.to(device), label_batch.to(device)).item()

            idx_split += 1

        sample_loss /= idx_split
        dataset_loss += sample_loss

        if len(entry_score_yes):
            pred_prob_yes = sum(entry_score_yes) / len(entry_score_yes)

            all_labels.append(label_bin)
            all_preds.append(pred_prob_yes)

    dataset_loss /= idx_sample + 1

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



def eval_model_index(
    model,
    dataloader=None,
    yes_idx=None,
    no_idx=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    loss_crt=None,
    get_logits_fun=None,
    max_batch_size=1,
):

    all_labels = []
    all_preds = []
    preds_acc = []

    chunk_labels = []

    dataset_loss = 0
    idx_sample = 0
    dataset_scores = []
    indices = []

    for idx_sample, ds_sample in enumerate(
        tqdm(dataloader, desc="eval")
    ):
        if isinstance(model, TrainablePromptModel):
            # To do
            warnings.warn("Unknown model, classification fallback.")
            (batch_iid, batch_am, batch_tti, doc_ids, labels) = ds_sample
        elif isinstance(model, TrainableClfModel):
            (batch_iid, batch_am, batch_tti, doc_ids, labels) = ds_sample
        else:
            warnings.warn("Unknown model, classification fallback.")
            (batch_iid, batch_am, batch_tti, doc_ids, labels) = ds_sample

        labels_bin = [1 if label.cpu().item() == yes_idx else 0 for label in labels]

        output = model(
            {
                "input_ids": batch_iid.to(device),
                "attention_mask": batch_am.to(device),
                "token_type_ids": batch_tti.to(device),
            }
        )[0]

        batch_size = output.shape[0]

        if isinstance(model, TrainablePromptModel):
            logits = get_logits_fun(output, batch_size, batch_mask_pos)
            label_batch = label.repeat(batch_size)
            logits_yes = logits[:, yes_idx]
            logits_no = logits[:, no_idx]

        elif isinstance(model, TrainableClfModel):
            logits = get_logits_fun(output)
            label_batch = torch.tensor(labels_bin)
            logits_yes = logits[:, 1]
            logits_no = logits[:, 0]
            
        logits_yes_no = torch.stack((logits_no, logits_yes), dim=1)

        probs_yes_no = softmax(logits_yes_no, dim=1)
        dataset_scores.extend(probs_yes_no[:, 1].cpu().numpy().tolist())
        indices.extend(doc_ids.cpu().numpy().tolist())
        chunk_labels.extend(labels_bin)
        sample_loss = loss_crt(logits.to(device), label_batch.to(device)).item()

        dataset_loss += sample_loss

        # if len(entry_score_yes):
        #     pred_prob_yes = sum(entry_score_yes) / len(entry_score_yes)

        #     all_labels.append(label_bin)
        #     all_preds.append(pred_prob_yes)

    dataset_loss /= (idx_sample + 1)

    n_samples = max(indices)

    all_preds = []
    all_labels = []
    dataset_scores = np.array(dataset_scores)
    indices = np.array(indices)
    chunk_labels = np.array(chunk_labels)

    for sample_idx in range(n_samples+1):
        all_preds.append(np.mean(dataset_scores[indices==sample_idx]))
        all_labels.append(chunk_labels[indices==sample_idx][0])

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
