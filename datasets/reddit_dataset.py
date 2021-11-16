from transformers import PreTrainedTokenizerBase
from torch.utils.data import Datasetq


class RedditDataset(Dataset):
    """
    This dataset should work on the Reddit minidataset.
    """
    def __init__(self, 
                path: str,
                tokenizer: PreTrainedTokenizerBase,
                debug: bool=False,
                padding_end: bool=False,
                train: bool=False,
                just_first_seq: bool=False,
                device: str='cuda'
        ):
        """
        Arguments:
            path: path to folder containing the .json files (or to a .jsonl file)
            tokenizer: Huggingface or other tokenizer with ```tokenize``` method
            debug: if True, load a few examples only
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
        sequence_length = int(self.block_size/2)
        self.mask_position = sequence_length + 2
        self.tokenizer = tokenizer
        self.yes_idx = torch.tensor(tokenizer.convert_tokens_to_ids("yes"))
        self.no_idx = torch.tensor(tokenizer.convert_tokens_to_ids("no"))
        self.debug = debug
        self.padding_end = padding_end
        self.train = train
        self.just_first_seq = just_first_seq        
        self.device = device

#         print('YES IDX:', self.yes_idx)
#         print('YES IDX:', self.no_idx)

        if self.folder_input:
            self.json_files = [fname for fname in os.listdir(path) if fname.endswith('.json')]
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

        sequence_length = int(self.block_size/2)

        sample_list = []
        mask_position = []
        
        len_s1 = len(sample1_tokens)
        len_s2 = len(sample2_tokens)
        
        if self.train:
            if self.padding_end:
                #fix bug
                if len_s1 > sequence_length:
                    if self.just_first_seq:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, len_s1 - sequence_length - 1)
                    sample1_tokens = sample1_tokens[start_idx: start_idx + sequence_length]
                else:
                    sample1_tokens.extend([pad_token] * (sequence_length - len_s1))
                    
                if len_s2 > sequence_length:
                    if self.just_first_seq:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, len_s1 - sequence_length - 1)
                    sample2_tokens = sample2_tokens[start_idx: start_idx + sequence_length]
                else:
                    sample2_tokens.extend([pad_token] * (sequence_length - len_s2))

                    
                entire_sequence = [cls_token] + sample1_tokens + [sep_token] + [mask_token] + sample2_tokens + [sep_token]
                padding_length = self.max_seq_length - len(entire_sequence)
                attention_mask =  [1] * len(entire_sequence) + [0] * padding_length
                entire_sequence += [pad_token] * padding_length
            else:
                if len_s1 > sequence_length:
                    if self.just_first_seq:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, len_s1 - sequence_length - 1)
                        
                    sample1_tokens = sample1_tokens[start_idx: start_idx + sequence_length]
                else:
                    sample1_tokens.extend([pad_token] * (sequence_length - len_s1))
                    
                if len_s2 > sequence_length:
                    if self.just_first_seq:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, len_s1 - sequence_length - 1)
                        
                    sample2_tokens = sample2_tokens[start_idx: start_idx + sequence_length]
                else:
                    sample2_tokens.extend([pad_token] * (sequence_length - len_s2))
                    
                entire_sequence = [cls_token] + sample1_tokens + [sep_token] + [mask_token] + sample2_tokens + [sep_token]

                    
            len_s1 = len(sample1_tokens)
            len_s2 = len(sample2_tokens)
            token_type_ids = [0 if idx_2 < (len_s1+2) else 1 for idx_2 in range(self.max_seq_length)]
            tokenized_seq = self.tokenizer.convert_tokens_to_ids(entire_sequence)
            attention_mask = [1 if t != 0 else 0 for t in tokenized_seq]
            
            assert len(entire_sequence) == len(attention_mask), 'uneven seqs'
            assert len(token_type_ids) == len(tokenized_seq), 'uneven seqs'

            label = self.yes_idx if entry['same'] else self.no_idx
            mask_position = list(tokenized_seq).index(103)
            return (
                {"input_ids": torch.tensor(tokenized_seq),
                 "token_type_ids": torch.tensor(token_type_ids),
                 "attention_mask": torch.tensor(attention_mask)
                }, 
                torch.tensor(label),
                torch.tensor(mask_position)
            )
        else:
            for idx_1 in range(0, min_size, sequence_length):
                seq1 = sample1_tokens[idx_1: idx_1+sequence_length]
                seq2 = sample2_tokens[idx_1: idx_1+sequence_length]

                len_s1 = len(seq1)
                len_s2 = len(seq2)

                if self.padding_end:
                    mask_position.append(len_s1 + 2)
                    entire_sequence = [cls_token] + seq1 + [sep_token] + [mask_token] + seq2 + [sep_token]
                    padding_length = self.max_seq_length - len(entire_sequence)
                    attention_mask = [1] * len(entire_sequence) + [0] * padding_length
                    entire_sequence += [pad_token] * padding_length
                    tokenized_seq = self.tokenizer.convert_tokens_to_ids(entire_sequence)
                else:
                    if len_s1 < sequence_length:
                        seq1.extend([pad_token] * (sequence_length - len_s1))

                    if len_s2 < sequence_length:
                        seq2.extend([pad_token] * (sequence_length - len_s2))

                    entire_sequence = [cls_token] + seq1 + [sep_token] + [mask_token] + seq2 + [sep_token]
                    tokenized_seq = self.tokenizer.convert_tokens_to_ids(entire_sequence)
                    mask_position.append(list(tokenized_seq).index(103))
                
                len_s1 = len(seq1)
                len_s2 = len(seq2)
                token_type_ids = [0 if idx_2 < (len_s1+2) else 1 for idx_2 in range(self.max_seq_length)]

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
                        "token_type_ids": torch.tensor(token_type_ids)
                    }
                )

#             label = [1 for _ in range(len(sample_list))] if entry['same'] else [0 for _ in range(len(sample_list)) ]
            label = [1] if entry['same'] else [0]
            
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