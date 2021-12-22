import argparse
import os
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from models import TrainableClfModel
from utils import train_model
from datasets import RedditClsDataset, RedditClsDataset_index
from torch.utils.tensorboard import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Prompt-based Training Script")

parser.add_argument(
    "--train_dir",
    type=str,
    default="/darkweb_ds/open_splits/unseen_all/xs/pan20-av-small-train.jsonl",
)

parser.add_argument(
    "--val_dir",
    type=str,
    default="/darkweb_ds/open_splits/unseen_all/xs/pan20-av-small-val.jsonl",
)
parser.add_argument(
    "--test_dir",
    type=str,
    default="/darkweb_ds/open_splits/unseen_all/xs/pan20-av-small-test.jsonl",
)
parser.add_argument(
    "--tb_dir",
    type=str,
    default="./cls_runs",
)
parser.add_argument(
    "--exp_prefix",
    type=str,
    default="XS_CLS_Openall",
)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--trainable_params", type=str, default="all")  # bias, linear

args = parser.parse_args()

train_dataset_path = args.train_dir
val_dataset_path = args.val_dir
test_dataset_path = args.test_dir
lr = args.lr
wd = args.wd
batch_size = args.batch_size
epochs = args.epochs
tb_dir = args.tb_dir
trainable_params = args.trainable_params
exp_prefix = (
    args.exp_prefix
    + f"_lr{lr}_wd{wd}_bs{batch_size}_trainable_params{trainable_params}"
)

best_model_path = os.path.join("./checkpoints", exp_prefix)

tb_writer = SummaryWriter(log_dir=os.path.join(tb_dir, exp_prefix))

model = TrainableClfModel()
if trainable_params == "bias":
    trainable_params = [p for (n, p) in model.model.named_parameters() if "bias" in n]
elif trainable_params == "linear":
    trainable_params = model.model.classifier.parameters()
else:
    no_decay = ["bias", "LayerNorm.weight"]
    trainable_params = [
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

train_dataset = RedditClsDataset(
    path=train_dataset_path,
    tokenizer=model.tokenizer,
    debug=False,
    train=True,
    load_data_to_memory=True,
)

train_dataset_full = RedditClsDataset(
    path=train_dataset_path,
    tokenizer=model.tokenizer,
    debug=False,
    train=False,
    load_data_to_memory=False,
)
val_dataset_full = RedditClsDataset_index(
    path=val_dataset_path, tokenizer=model.tokenizer, debug=False, train=False
)
test_dataset_full = RedditClsDataset_index(
    path=test_dataset_path,
    tokenizer=model.tokenizer,
    debug=False,
    train=False,
)

train_sampler = RandomSampler(train_dataset)
train_sampler_full = SequentialSampler(train_dataset_full)
val_sampler_full = SequentialSampler(val_dataset_full)
test_sampler_full = SequentialSampler(test_dataset_full)


train_dataloader = DataLoader(
    train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=16
)

train_dataloader_full = DataLoader(
    train_dataset_full, sampler=train_sampler_full, batch_size=1
)

val_dataloader_full = DataLoader(
    val_dataset_full, sampler=val_sampler_full, batch_size=batch_size, num_workers=16
)

test_dataloader_full = DataLoader(
    test_dataset_full, sampler=test_sampler_full, batch_size=batch_size, num_workers=16
)

optimizer = AdamW(trainable_params, lr=lr, weight_decay=wd)
loss_crt = CrossEntropyLoss()

train_model(
    model=model,
    optimizer=optimizer,
    loss_crt=loss_crt,
    train_dataloader=train_dataloader,
    train_dataloader_full=train_dataloader_full,
    val_dataloader_full=val_dataloader_full,
    test_dataloader_full=test_dataloader_full,
    scheduler=ReduceLROnPlateau,
    epochs=epochs,
    tb_writer=tb_writer,
    best_model_path=best_model_path,
)
