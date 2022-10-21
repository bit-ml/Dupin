from collections import defaultdict
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import json
import uuid
import os
import random

def get_dataset_token_count(dataset):
    count = 0
    for entry in dataset:
        count = count + len(entry['pair'][0].split()) + len(entry['pair'][1].split())

    return count

def make_sa_pair(comment: List[str]) -> dict:
    comment_size = len(comment)
    return {
        "id": str(uuid.uuid4()),
        "pair": [" ".join(comment[:comment_size//2]), 
                 " ".join(comment[comment_size//2:])
        ],
        "same": True
    }

def get_closed_split(author_comments) -> Tuple[List[dict]]:
    """
    Split the dataset in an closed setup, where authors in 
    test/dev are seen in training examples as well
    Arguments: 
        author_comments: dictionary where author name is the key and 
                         his comments (tokenizer) is the value

    Returns a list of dictionaries for each train/val/test split.
    """
    # authors with most comments in ascending order
    author_len = {a:len(c) for a,c in author_comments.items()}
    print("Number of authors: ", len(author_len))
    print("Number of comments: ", len(data["comment_body"]))
    print("Average number of tokens per author: ", sum(author_len.values())/len(author_len))
    
    # keep top authors
    top_authors = sorted([(a,l) for a,l in author_len.items() if l>=768], key=lambda x:x[1], reverse=True)
    print("Number of verbose authors (>= 768 tokens): ", len(top_authors))
    print("Verbose authors: (author, #num_tokens): ", top_authors)
    top_authors_name = [t[0] for t in top_authors]

    top_authors_comments = {a:c for a,c in author_comments.items() if a in top_authors_name}
    print('================================================')
    print("#top_authors / #authors = ", len(top_authors), "/", len(author_comments))
    print('================================================')

    # iterate through all authors
    # split author's text into chunks of 256 tokens:
    #   | c1 | c2 | c3 | c4 | ... | cN|
    # To make same-author pairs, each chunk ci is split in two sequences
    # 
    # Chunk allocations:
    #   - 3 chunks, split them into 1 (train), 1 (dev), 1 (test)
    #   - 4 chunks, split them into 1 (train), 1 (dev), 2 (test)
    #   - >=5 chunks, split them into 20% (train), 40% (dev), 40% (test)
    #
    # To make different-author pairs, take 2 authors and their chunks:
    # Build one example by sampling a different chunk from each author

    # maps author to list with chunks (where the chunk is a list of tokens)
    author_chunks = {} 

    # make SA pairs
    train_dataset, val_dataset, test_dataset = [], [], []
    for a, comments in top_authors_comments.items():
        num_chunks = len(comments) // 256
        comment_chunks = [comments[i*256:(i+1)*256] for i in range(num_chunks)]
        author_chunks[a] = comment_chunks
        
        sa_pairs = [make_sa_pair(comment_chunks[i]) for i in range(num_chunks)]
        if num_chunks == 3:
            train_dataset.append(sa_pairs[0])
            val_dataset.append(sa_pairs[1])
            test_dataset.append(sa_pairs[2])
        elif num_chunks == 4:
            train_dataset.append(sa_pairs[0])
            val_dataset.append(sa_pairs[1])
            test_dataset += [sa_pairs[2], sa_pairs[3]]
        else:
            train_examples, other = train_test_split(sa_pairs, train_size=0.2)
            val_examples, test_examples = train_test_split(other, train_size=0.5)
            train_dataset += train_examples
            val_dataset += val_examples
            test_dataset += test_examples

            print("num_chunks = %d: %d (train), %d (val), %d (test)" % (
                num_chunks, len(train_examples), len(val_examples), len(test_examples)))

    # make DA pairs
    author_pairs = [(a1, a2) for a1 in top_authors_name for a2 in top_authors_name if a1 != a2]

    num_da_train_pairs = len(train_dataset)
    num_da_val_pairs = len(val_dataset)
    num_da_test_pairs = len(test_dataset)

    da_pairs_train = random.sample(author_pairs, num_da_train_pairs)
    da_pairs_val = random.sample(author_pairs, num_da_val_pairs)
    da_pairs_test = random.sample(author_pairs, num_da_test_pairs)
    
    da_pairs = [da_pairs_train, da_pairs_val, da_pairs_test]
    datasets = [train_dataset, val_dataset, test_dataset]

    for (ds, da_pairs) in zip(datasets, da_pairs):
        for (ai, aj) in da_pairs:
            ai_comments = top_authors_comments[ai]
            aj_comments = top_authors_comments[aj]
            ai_len = len(ai_comments)
            aj_len = len(aj_comments)
            # select 128-length random comments from each author
            start_idx = random.randint(0, ai_len-128)
            ci = ai_comments[start_idx: start_idx+128]
            start_idx = random.randint(0, aj_len-128)
            cj = aj_comments[start_idx: start_idx+128]

            ds.append({
                "id": str(uuid.uuid4()),
                "pair": [" ".join(ci), " ".join(cj)],
                "same": False
            })
    
    return (train_dataset, val_dataset, test_dataset)

def get_open_split(author_comments) -> Tuple[List[dict]]:
    """
    Split the dataset in an open setting, unknown authors.
    Arguments: 
        author_comments: dictionary where author name is the key and 
                         his comments (tokenizer) is the value

    Returns a list of dictionaries for each train/val/test split.
    """
    # authors with most comments in ascending order
    author_len = {a:len(c) for a,c in author_comments.items()}
    print("Number of authors: ", len(author_len))
    print("Number of comments: ", len(data["comment_body"]))
    print("Average number of tokens per author: ", sum(author_len.values())/len(author_len))
    
    # keep top authors
    top_authors = sorted([(a,l) for a,l in author_len.items() if l>=256], key=lambda x:x[1], reverse=True)
    print("Number of verbose authors (>= 256 tokens): ", len(top_authors))
    print("Verbose authors: (author, #num_tokens): ", top_authors)
    top_authors_name = [t[0] for t in top_authors]

    top_authors_comments = {a:c for a,c in author_comments.items() if a in top_authors_name}
    print('================================================')
    print("#top_authors / #authors = ", len(top_authors), "/", len(author_comments))
    print('================================================')

    # create disjoint author lists for train/val/test
    train_authors, test_authors = train_test_split(top_authors_name, test_size = 0.8)
    val_authors, test_authors = train_test_split(test_authors, test_size = 0.5)
    
    # sanity checks
    train_tokens, val_tokens, test_tokens = 0, 0, 0
    train_tokens = sum([len(c) for (a,c) in top_authors_comments.items() if a in train_authors])
    val_tokens = sum([len(c) for (a,c) in top_authors_comments.items() if a in val_authors])
    test_tokens = sum([len(c) for (a,c) in top_authors_comments.items() if a in test_authors])
    #print("Number of tokens: train (%d), val (%d), test (%d)" % (train_tokens, val_tokens, test_tokens))

    # make SA pairs (open split - disjoint authors)
    train_dataset, val_dataset, test_dataset = [], [], []
    dataset = []
    for a, c in top_authors_comments.items():
        comment_size = len(c)
        if a in train_authors:
            dataset = train_dataset
        elif a in val_authors:
            dataset = val_dataset
        else:
            dataset = test_dataset

        dataset.append({
            "id": str(uuid.uuid4()),
            "pair": [" ".join(c[:comment_size//2]), " ".join(c[comment_size//2:])],
            "same": True
        })

    # make DA pairs
    train_author_pairs = [(a1, a2) for a1 in train_authors for a2 in train_authors if a1 != a2]
    val_author_pairs = [(a1, a2) for a1 in val_authors for a2 in val_authors if a1 != a2]
    test_author_pairs = [(a1, a2) for a1 in test_authors for a2 in test_authors if a1 != a2]
    num_da_train_pairs = len(train_dataset)
    num_da_val_pairs = len(val_dataset)
    num_da_test_pairs = len(test_dataset)

    combo = zip(
        [train_author_pairs, val_author_pairs, test_author_pairs],
        [num_da_train_pairs, num_da_val_pairs, num_da_test_pairs],
        [train_dataset, val_dataset, test_dataset]
    )
    for (split, size, ds) in combo:
        da_pairs = random.sample(split, size)
        for a1, a2 in da_pairs:
            c1, c2 = top_authors_comments[a1], top_authors_comments[a2]
            ds.append({
                "id": str(uuid.uuid4()),
                "pair": [" ".join(c1), " ".join(c2)],
                "same": False
            })
    
    return (train_dataset, val_dataset, test_dataset)

if __name__ == '__main__':
    comment_fn = '/pan2020/reddit_darknet/darknetCommentData.json'
    ds_type = "closed"

    with open(comment_fn, "r") as fp:
        data = json.load(fp)
        print("Number of authors: ", len(data["author_names"]))
        print("Number of comments: ", len(data["comment_body"]))

    author_comments = defaultdict(list)
    for idx, (a,c) in enumerate(zip(data['author_names'], data['comment_body'])):
        # skip bot users or moderators
        if 'bot' in a or 'Bot' in a or a == 'AutoModerator':
            continue
        if idx % 100 == 0:
            print("Processed ", idx, " comments")
            #break
        author_comments[a] += c.split()

    if ds_type == 'open':
        datasets = get_open_split(author_comments)
    else:
        datasets = get_closed_split(author_comments)

    #datasets =  [train_dataset, val_dataset, test_dataset]
    for (ds_name, ds) in zip(['train', 'val', 'test'], datasets):
        l = len(ds)
        print('%s: %d examples, %d tokens' % (ds_name, len(ds), get_dataset_token_count(ds)))
        print("  %d SA tokens, %d DA tokens" % (
            get_dataset_token_count(ds[:l//2]), 
            get_dataset_token_count(ds[l//2:]))
        )

    train_len, val_len, test_len = len(datasets[0]), len(datasets[1]), len(datasets[2])
    random.shuffle(datasets[0])
    random.shuffle(datasets[1])
    random.shuffle(datasets[2])

    train_path = "/pan2020/reddit_darknet/reddit_%s_split/train" % (ds_type)
    val_path = "/pan2020/reddit_darknet/reddit_%s_split/val" % (ds_type)
    test_path = "/pan2020/reddit_darknet/reddit_%s_split/test" % (ds_type)
    splits = ["train", "val", "test"]

    # create intermmediate splits
    perc10, perc20, perc50 = (int)(train_len*0.1), (int)(train_len*0.2), (int)(train_len*0.5) 
    perc_lens = [perc10, perc20, perc50]
    perc10_path = "/pan2020/reddit_darknet/reddit_%s_split_10per" % (ds_type)
    perc20_path = "/pan2020/reddit_darknet/reddit_%s_split_20per" % (ds_type)
    perc50_path = "/pan2020/reddit_darknet/reddit_%s_split_50per" % (ds_type)
    perc_paths = [perc10_path, perc20_path, perc50_path]

    # write train/val/test datasets for each 10/20/50 percent variants
    for (perc_len, perc_path) in zip(perc_lens, perc_paths):
        for idx, (split, ds) in enumerate(zip(splits, datasets)):
            folder_name = os.path.join(perc_path, split)
            # if train, resize dataset
            ds_size = perc_len if idx == 0 else len(ds)
            for entry in ds[:ds_size]:
                fname = entry['id'] + ".json"
                with open(os.path.join(folder_name, fname), "w") as fp:
                    json.dump(entry, fp, indent=2)

    # write full dataset
    for (ds_path, ds) in zip([train_path, val_path, test_path], datasets):
        for entry in ds:
            fname = entry['id'] + ".json"
            with open(os.path.join(ds_path, fname), "w") as fp:
                json.dump(entry, fp, indent=2)

    
    