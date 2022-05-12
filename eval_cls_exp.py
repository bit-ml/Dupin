import argparse
import os
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from transformers import AdamW
import torch
from models import TrainableClfModel
from utils import train_model
from utils.train_utils import eval_model_index, get_logits_clf_eval, eval_model
from datasets import RedditClsDataset, RedditClsDataset_index
from torch.utils.tensorboard import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Prompt-based Training Script")

parser.add_argument(
    "--train_dir",
    type=str,
    default="/pan2020/open_splits/unseen_all/xs/pan20-av-small-train.jsonl",
)

parser.add_argument(
    "--val_dir",
    type=str,
    default="/pan2020/open_splits/unseen_all/xs/pan20-av-small-val.jsonl",
)
parser.add_argument(
    "--test_dir",
    type=str,
    default="/pan2020/open_splits/unseen_all/xs/pan20-av-small-test.jsonl",
)
parser.add_argument(
    "--tb_dir",
    type=str,
    default="./cls_runs",
)
parser.add_argument(
    "--exp_prefix",
    type=str,
    default="Junk_XS_CLS_Openall",
)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--trainable_params", type=str, default="linear")  # bias, linear

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

MODEL_NAMES = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-base-v2'
}

model = TrainableClfModel(model_name=MODEL_NAMES["distilbert"])#, device=torch.device('cpu'))

if trainable_params == "bias":
    trainable_params = [p for (n, p) in model.model.named_parameters() if "bias" in n]
elif trainable_params == "linear":
    trainable_params = list(model.model.classifier.parameters())
else:
    no_decay = ["bias", "LayerNorm.weight"]
    trainable_params = [
        {
            "params": [
                p
                for n, p in model.model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": wd,
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

# for p in trainable_params:
#     print(p)

for p in model.model.parameters():
    p.requires_grad = False

for p in trainable_params:
    p.requires_grad = True

# for (n, p) in model.model.named_parameters():
#     print((n,p))

optimizer = AdamW(trainable_params, lr=lr, weight_decay=wd)


test_dataset_full = RedditClsDataset_index(
    path=test_dataset_path,
    tokenizer=model.tokenizer,
    debug=False,
    train=False,
)

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    return size_all_mb

print('model size: {:.3f}MB'.format(get_model_size(model.model)))

#print("dataset example 0 = ", test_dataset_full[0])

test_sampler_full = SequentialSampler(test_dataset_full)


test_dataloader_full = DataLoader(
    test_dataset_full, sampler=test_sampler_full, batch_size=batch_size, num_workers=16
)

loss_crt = CrossEntropyLoss()

results = eval_model_index(
    model,
    dataloader=test_dataloader_full,
    yes_idx=None,
    no_idx=None,
    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    device=torch.device("cpu"),
    loss_crt=loss_crt,
    get_logits_fun=get_logits_clf_eval,
    max_batch_size=32,
)

print('results = ', results)
# train_model(
#     model=model,
#     optimizer=optimizer,
#     loss_crt=loss_crt,
#     train_dataloader=train_dataloader,
#     train_dataloader_full=train_dataloader_full,
#     val_dataloader_full=val_dataloader_full,
#     test_dataloader_full=test_dataloader_full,
#     scheduler=ReduceLROnPlateau,
#     epochs=epochs,
#     tb_writer=tb_writer,
#     best_model_path=best_model_path,
# )
