from transformers import PreTrainedTokenizerBase, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from typing import Dict
import os
import json
import random


class RedditPromptDataset(Dataset):
    """
    Dataset used to train prompting methods.
    The null prompt is encoded differently, depending on the mask_placement
    value. By default, the [MASK] token separates the two sequences:
        [CLS] S1 [SEP] [MASK] S2 [SEP]
    """

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        debug: bool = False,
        padding_end: bool = False,
        train: bool = False,
        just_first_seq: bool = False,
        mask_placement: str = "middle",
        device: str = "cuda",
    ):
        """
        Arguments:
            path: path to folder containing the .json files (or to a .jsonl file)
            tokenizer: Huggingface or other tokenizer with ```tokenize``` method
            debug: if True, load a few examples only
            padding_end: if True, null prompt is encoded as:
                [CLS] S1 [SEP] [MASK] S2 [SEP] [PAD] ... [PAD]
                    if False, padding is added after each sequence if possible:
                [CLS] S1 [PAD] ... [PAD] [SEP] [MASK] S2 [SEP] [PAD] ... [PAD]
            mask_placement: where to put the [MASK] token inside the null prompt:
                'beginning': [CLS] [MASK] S1 [SEP] S2 [SEP] [PAD] ... [PAD]
                'middle':    [CLS] S1 [SEP] [MASK] S2 [SEP] [PAD] ... [PAD]
                'end':       [CLS] S1 [SEP] S2 [SEP] [MASK] [PAD] ... [PAD]

        """
        if not os.path.exists(path):
            print("inexistent path ", path)
            raise ValueError

        if path.endswith(".jsonl"):
            self.folder_input = False
        else:
            self.folder_input = True

        self.path = path
        self.max_seq_length = 512
        self.block_size = self.max_seq_length - 4
        sequence_length = int(self.block_size / 2)
        self.mask_position = sequence_length + 2
        self.tokenizer = tokenizer
        self.yes_idx = torch.tensor(tokenizer.convert_tokens_to_ids("yes"))
        self.no_idx = torch.tensor(tokenizer.convert_tokens_to_ids("no"))
        self.debug = debug
        self.padding_end = padding_end
        self.train = train
        self.just_first_seq = just_first_seq
        self.device = device
        self.mask_placement = mask_placement

        if self.folder_input:
            self.json_files = [
                fname for fname in os.listdir(path) if fname.endswith(".json")
            ]
        else:
            self.json_files = []
            with open(path) as fp:
                for line in fp.readlines():
                    self.json_files.append(json.loads(line))

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        sep_token = self.tokenizer.sep_token
        cls_token = self.tokenizer.cls_token
        pad_token = self.tokenizer.pad_token
        mask_token = self.tokenizer.mask_token

        if self.folder_input:
            json_file = os.path.join(self.path, self.json_files[idx])
            with open(json_file) as fp:
                entry = json.load(fp)
        else:
            entry = self.json_files[idx]

        sample1_tokens = self.tokenizer.tokenize(entry["pair"][0])
        sample2_tokens = self.tokenizer.tokenize(entry["pair"][1])

        len_s1 = len(sample1_tokens)
        len_s2 = len(sample2_tokens)

        min_size = min(len_s1, len_s2)

        sample1_tokens = sample1_tokens[:min_size]
        sample2_tokens = sample2_tokens[:min_size]

        sequence_length = int(self.block_size / 2)

        sample_list = []
        mask_position = []

        len_s1 = len(sample1_tokens)
        len_s2 = len(sample2_tokens)

        if self.train:
            if self.padding_end:
                # fix bug
                if len_s1 > sequence_length:
                    if self.just_first_seq:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, len_s1 - sequence_length - 1)
                    sample1_tokens = sample1_tokens[
                        start_idx : start_idx + sequence_length
                    ]
                else:
                    sample1_tokens.extend([pad_token] * (sequence_length - len_s1))

                if len_s2 > sequence_length:
                    if self.just_first_seq:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, len_s1 - sequence_length - 1)
                    sample2_tokens = sample2_tokens[
                        start_idx : start_idx + sequence_length
                    ]
                else:
                    sample2_tokens.extend([pad_token] * (sequence_length - len_s2))

                if self.mask_placement == "beginning":
                    entire_sequence = (
                        [cls_token]
                        + [mask_token]
                        + sample1_tokens
                        + [sep_token]
                        + sample2_tokens
                        + [sep_token]
                    )
                elif self.mask_placement == "middle":
                    entire_sequence = (
                        [cls_token]
                        + sample1_tokens
                        + [sep_token]
                        + [mask_token]
                        + sample2_tokens
                        + [sep_token]
                    )
                else:
                    entire_sequence = (
                        [cls_token]
                        + sample1_tokens
                        + [sep_token]
                        + sample2_tokens
                        + [sep_token]
                        + [mask_token]
                    )
                padding_length = self.max_seq_length - len(entire_sequence)
                attention_mask = [1] * len(entire_sequence) + [0] * padding_length
                entire_sequence += [pad_token] * padding_length
            else:
                if len_s1 > sequence_length:
                    if self.just_first_seq:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, len_s1 - sequence_length - 1)

                    sample1_tokens = sample1_tokens[
                        start_idx : start_idx + sequence_length
                    ]
                else:
                    sample1_tokens.extend([pad_token] * (sequence_length - len_s1))

                if len_s2 > sequence_length:
                    if self.just_first_seq:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, len_s1 - sequence_length - 1)

                    sample2_tokens = sample2_tokens[
                        start_idx : start_idx + sequence_length
                    ]
                else:
                    sample2_tokens.extend([pad_token] * (sequence_length - len_s2))

                if self.mask_placement == "beginning":
                    entire_sequence = (
                        [cls_token]
                        + [mask_token]
                        + sample1_tokens
                        + [sep_token]
                        + sample2_tokens
                        + [sep_token]
                    )
                elif self.mask_placement == "middle":
                    entire_sequence = (
                        [cls_token]
                        + sample1_tokens
                        + [sep_token]
                        + [mask_token]
                        + sample2_tokens
                        + [sep_token]
                    )
                else:
                    entire_sequence = (
                        [cls_token]
                        + sample1_tokens
                        + [sep_token]
                        + sample2_tokens
                        + [sep_token]
                        + [mask_token]
                    )

            len_s1 = len(sample1_tokens)
            len_s2 = len(sample2_tokens)
            if self.mask_placement == "beginning":
                token_type_ids = [
                    0 if idx_2 < (len_s1 + 3) else 1
                    for idx_2 in range(self.max_seq_length)
                ]
            else:
                token_type_ids = [
                    0 if idx_2 < (len_s1 + 2) else 1
                    for idx_2 in range(self.max_seq_length)
                ]
            tokenized_seq = self.tokenizer.convert_tokens_to_ids(entire_sequence)
            attention_mask = [1 if t != 0 else 0 for t in tokenized_seq]

            assert len(entire_sequence) == len(attention_mask), "uneven seqs"
            assert len(token_type_ids) == len(tokenized_seq), "uneven seqs"

            label = self.yes_idx if entry["same"] else self.no_idx
            mask_position = list(tokenized_seq).index(103)
            return (
                {
                    "input_ids": torch.tensor(tokenized_seq),
                    "token_type_ids": torch.tensor(token_type_ids),
                    "attention_mask": torch.tensor(attention_mask),
                },
                torch.tensor(label),
                torch.tensor(mask_position),
            )
        else:
            for idx_1 in range(0, min_size, sequence_length):
                seq1 = sample1_tokens[idx_1 : idx_1 + sequence_length]
                seq2 = sample2_tokens[idx_1 : idx_1 + sequence_length]

                len_s1 = len(seq1)
                len_s2 = len(seq2)

                if self.padding_end:
                    mask_position.append(len_s1 + 2)
                    if self.mask_placement == "beginning":
                        entire_sequence = (
                            [cls_token]
                            + [mask_token]
                            + seq1
                            + [sep_token]
                            + seq2
                            + [sep_token]
                        )
                    elif self.mask_placement == "middle":
                        entire_sequence = (
                            [cls_token]
                            + seq1
                            + [sep_token]
                            + [mask_token]
                            + seq2
                            + [sep_token]
                        )
                    else:
                        entire_sequence = (
                            [cls_token]
                            + seq1
                            + [sep_token]
                            + seq2
                            + [sep_token]
                            + [mask_token]
                        )

                    padding_length = self.max_seq_length - len(entire_sequence)
                    attention_mask = [1] * len(entire_sequence) + [0] * padding_length
                    entire_sequence += [pad_token] * padding_length
                    tokenized_seq = self.tokenizer.convert_tokens_to_ids(
                        entire_sequence
                    )
                else:
                    if len_s1 < sequence_length:
                        seq1.extend([pad_token] * (sequence_length - len_s1))

                    if len_s2 < sequence_length:
                        seq2.extend([pad_token] * (sequence_length - len_s2))

                    if self.mask_placement == "beginning":
                        entire_sequence = (
                            [cls_token]
                            + [mask_token]
                            + seq1
                            + [sep_token]
                            + seq2
                            + [sep_token]
                        )
                    elif self.mask_placement == "middle":
                        entire_sequence = (
                            [cls_token]
                            + seq1
                            + [sep_token]
                            + [mask_token]
                            + seq2
                            + [sep_token]
                        )
                    else:
                        entire_sequence = (
                            [cls_token]
                            + seq1
                            + [sep_token]
                            + seq2
                            + [sep_token]
                            + [mask_token]
                        )
                    tokenized_seq = self.tokenizer.convert_tokens_to_ids(
                        entire_sequence
                    )
                    mask_position.append(list(tokenized_seq).index(103))

                len_s1 = len(seq1)
                len_s2 = len(seq2)
                if self.mask_placement == "beginning":
                    token_type_ids = [
                        0 if idx_2 < (len_s1 + 3) else 1
                        for idx_2 in range(self.max_seq_length)
                    ]
                else:
                    token_type_ids = [
                        0 if idx_2 < (len_s1 + 2) else 1
                        for idx_2 in range(self.max_seq_length)
                    ]

                attention_mask = [1 if t != 0 else 0 for t in tokenized_seq]

                #                 print('len(seq1)', len(seq1))
                #                 print('len(seq2)', len(seq2))
                #                 print('len(tokenized_seq)', len(tokenized_seq))
                #                 print('len(token_type_ids)', len(token_type_ids))
                #                 print('len(attention_mask)', len(attention_mask))

                sample_list.append(
                    {
                        "input_ids": torch.tensor(tokenized_seq),
                        "attention_mask": torch.tensor(attention_mask),
                        "token_type_ids": torch.tensor(token_type_ids),
                    }
                )

            #             label = [1 for _ in range(len(sample_list))] if entry['same'] else [0 for _ in range(len(sample_list)) ]
            label = self.yes_idx if entry["same"] else self.no_idx

            label = torch.tensor(label)
            mask_position = torch.tensor(mask_position)
#             print('label.shape', label.shape)
#             print('mask_position.shape', mask_position.shape)
#             print('sample_list', len(sample_list))
            
#             for t in sample_list:
#                 print('t.input_ids', t["input_ids"].shape)
#                 print('t.attention', t["attention_mask"].shape)
#                 print('t.ttids', t["token_type_ids"].shape)
#                 print('\n')
#             print('\n-----------------------------')
            
            return sample_list, label, mask_position


def create_dataloader(**kwargs):
    dataset = RedditPromptDataset(**kwargs)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1
    )

    return loader

def group_duplicates_list_as_string(l):
    """
        Given a list of number/strings, summarize list by grouping
        consecutive entries. For instance
            [CLS] [MASK] [TXT] [TXT] [SEP] [TXT] [TXT] [TXT] [SEP] [PAD] ... [PAD]
        will be formatted as the following string:
            "[CLS] [MASK] 2x[TXT] [SEP] 3x[TXT] 40x[PAD]"
    """
    l_str = ""
    previous_entry = l[0]
    counter = 1
    for elem in l[1:]:
        if elem != previous_entry:
            if counter == 1:
                l_str += ' {0}'.format(previous_entry)
            else:
                l_str += ' {0}*{1}'.format(counter, previous_entry)
            previous_entry = elem
            counter = 1
        else:
            counter += 1
    if counter == 1:
        l_str += ' {0}'.format(previous_entry)
    else:
        l_str += ' {0}*{1}'.format(counter, previous_entry)       

    return l_str.strip()

def summarize_entry(entry: Dict):
    """
    Formats dataset entry in a human-readable form:
        token_ids:  "[CLS] 254*[TXT] [SEP] 254*[TXT] [SEP] [MASK]"
        token_type_ids: "256*0 256*1"
        attention_mask: "512*1"
    """
    special_ids = {
        0: '[PAD]',
        101: '[CLS]',
        102: '[SEP]',
        103: '[MASK]'
    }
    input_ids = entry['input_ids'][0].tolist()
    input_ids_txt = [special_ids[token_id] if token_id in special_ids else '[TXT]' for token_id in input_ids]
    input_ids_str =  group_duplicates_list_as_string(input_ids_txt)
    
    token_type_ids = entry['token_type_ids'][0]
    token_type_ids_str = group_duplicates_list_as_string(token_type_ids)

    attention_mask = entry['attention_mask'][0]
    attention_mask_str = group_duplicates_list_as_string(attention_mask)

    return input_ids_str, token_type_ids_str, attention_mask_str 


if __name__ == '__main__':
    example_path = "/pan2020/reddit_darknet/train/0004e99b-d8a2-4bb5-b3f6-f38309ca80af.json"
    dataset_path = "/pan2020/reddit_darknet/train"
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    for padding_end in [True, False]:
        for mask_placement in ['beginning', 'middle', 'end']:
            print("Setup: padding_end={0} mask={1}".format(padding_end, mask_placement))
            loader = create_dataloader(
                path=dataset_path,
                tokenizer=tokenizer,
                debug=False,
                padding_end=padding_end,
                train=False,
                just_first_seq=True,
                mask_placement=mask_placement,
                device='cuda'
            )

            for ex in loader:                
                token_ids_str, token_type_ids_str, attention_str = summarize_entry(ex[0][0])
                print("\tFirst batch: ")
                print("\t\ttoken_ids: ", token_ids_str)
                print("\t\ttoken_type_ids: ", token_type_ids_str)
                print("\t\tattention_mask: ", attention_str)
                
                print("\tLast batch: ")
                token_ids_str, token_type_ids_str, attention_str = summarize_entry(ex[0][-1])
                print("\t\ttoken_ids: ", token_ids_str)
                print("\t\ttoken_type_ids: ", token_type_ids_str)
                print("\t\tattention_mask: ", attention_str)
                break
