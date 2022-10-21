import json
import os
import random
import uuid
import numpy as np
from collections import defaultdict
from typing import Dict, Union, List, Callable, Tuple
from transformers import BertTokenizer, BertTokenizerFast

def merge_data_and_labels(data_path: str, ground_truth_path: str, merged_data_path: str):
    """
    Merge PAN authors data and ground truth data into a single ```.jsonl``` file.
    """
    # read ground truth data
    with open(ground_truth_path) as f:
        labels = [json.loads(line) for line in f]
    
    out_fp = open(merged_data_path, "w")

    with open(data_path) as f:
        for idx, line in enumerate(f):
            if idx % 100 == 0:
                print(idx)
            entry = json.loads(line)
            assert entry["id"] == labels[idx]["id"], "id mismatch at idx = %d" % (idx)
            entry["same"] = labels[idx]["same"]
            entry["authors"] = labels[idx]["authors"]
            json.dump(entry, out_fp)
            out_fp.write('\n')

    out_fp.close()


def remove_text_from_jsonl(input_jsonl: str, output_jsonl: str):
    """
    Remove text from .jsonl to get smaller files.
    Args:
        input_jsonl (str): path to original .jsonl file
        output_jsonl (str): path to new .jsonl file stripped of text 
    """
    with open(input_jsonl) as in_fp, open(output_jsonl, "w") as out_fp:
        for idx, line in enumerate(in_fp):
            if idx % 1000 == 0:
                print("idx = ", idx)
            entry = json.loads(line)
            del entry['pair']
            json.dump(entry, out_fp)
            out_fp.write('\n')

def write_jsonl_to_folder(path_to_jsonl: str, output_folder):
    """
    Store each JSON line in a .jsonl file in a separate .json file in ```output_folder```.
    """
    if not os.path.exists(output_folder):
        print(output_folder, " folder does not exist, trying to create it")
        os.mkdir(output_folder)
    
    if len(os.listdir(output_folder)) > 0:
        print(output_folder, " is not empty, abort")
        #print(os.listdir(output_folder))
        #return

    duplicate_count = 0
    with open(path_to_jsonl) as fp:
        for idx, line in enumerate(fp):
            if idx % 1000 == 0:
                print("[write_jsonl_to_folder] wrote %d .json files to %s, duplicates = %d" % (idx, output_folder, duplicate_count))
            entry = json.loads(line)
            entry_path = os.path.join(output_folder, entry["id"] + ".json")
            if os.path.exists(entry_path):
                duplicate_count += 1
            else:
                with open(entry_path, "w") as out_fp:
                    json.dump(entry, out_fp)
                    out_fp.write("\n")

    print("Duplicates: ", duplicate_count)


def read_jsonl_examples(data_path: str, count: int = -1) -> List[Dict]:
    """
    Reads example pairs from a ```.jsonl``` file and returns them as a list of dictionaries.
    If ```count``` is >= 0, try to read an equal number of same-author and
    different-author pairs.

    Args:
        data_path (str): path to .jsonl file containing the documents pairs, one per line
        count: how many examples to read; if -1, read all of them
    Returns:
        List[Dict]: the dataset
    """
    examples = []
    sa_count = count/2 if count > 0 else 0
    da_count = count-sa_count if count > 0 else 0
    with open(data_path) as fp:
        for idx, line in enumerate(fp):
            if idx % 10000 == 0:
                print("[read_jsonl_examples] read %d examples" % (idx))
            examples.append(json.loads(line))
            if idx == count:
                break
            # if example['same'] and sa_count > 0:
            #     examples.append(example)
            #     sa_count -= 1
            # elif not example['same'] and da_count > 0:
            #     examples.append(example)
            #     da_count -= 1

            # if sa_count == 0 and da_count == 0 and count >= 0:
            #     break

    return examples

def get_authors_data_from_folder(authors_folder: str, num_authors: int=-1) -> Dict:
    """
    Read authors from folder into a dictionary:
        {"$author_id": [{"fandom" : ..., "text": ....}]}
    """
    authors_data = defaultdict(list)
    author_files = os.listdir(authors_folder)
    if num_authors > 0:
        author_files = author_files[:num_authors]

    for idx, author_file in enumerate(author_files):
        if idx % 10000 == 0:
            print("[get_authors_data_from_folder] processed ", idx, " authors")
        author_id = author_file[:-6]
        with open(os.path.join(authors_folder, author_file)) as fp:
            author_data = []
            for line in fp:
                author_data.append(json.loads(line))
        authors_data[author_id] = author_data

    return authors_data

def get_authors_data_from_jsonl(path_to_jsonl: str, pan_authors_folder: str):
    """
    Get authors and their documents from the .jsonl doc pairs.
    For each author, save its documents in ```$author_id.jsonl``` inside
    the ```pan_authors_folder```. 
    Args:
        path_to_jsonl (str): [description]
        pan_authors_folder (str): [description]
    """
    authors_to_docs = defaultdict(list)
    with open(path_to_jsonl) as fp:
        for idx, line in enumerate(fp):
            if idx % 1000 == 0:
                print(idx)
            entry = json.loads(line)

            a1, a2 = entry['authors'][0], entry['authors'][1]
            d1, d2 = entry['pair'][0], entry['pair'][1]
            f1, f2 = entry['fandoms'][0], entry['fandoms'][1]
            for (a,f,d) in zip([a1, a2], [f1,f2], [d1, d2]):
                duplicate = False
                for entry in authors_to_docs[a]:
                    if entry['text'] == d:
                        duplicate = True
                        break
                if not duplicate:
                    authors_to_docs[a].append({'fandom': f, 'text': d})

    # write author data to ```pan_authors_folder```
    for author, docs in authors_to_docs.items():
        author_path = os.path.join(pan_authors_folder, author + ".jsonl")
        with open(author_path, "w") as fp:
            for doc in docs:
                json.dump(doc, fp)
                fp.write('\n')


def split_pan_dataset_closed_set_v1(examples: List[Dict], 
                                    test_split_percentage: float) -> (List, List):
    """
    Split data intro train and val/test in an almost closed-set fashion, by 
    following Araujo-Pino et al. 2020 
    ( https://pan.webis.de/downloads/publications/papers/araujopino_2020.pdf )
    This algorithm ensures that authors of SA pairs in the test set appear in the
    training set, but gives no guarantee that authors of test DA pairs appear in 
    the training set.
    Args:
        examples: dataset samples read from a .jsonl file using read_jsonl_examples()
        test_split_percentage: size of the Test split as a percentage of the whole dataset
    """
    assert test_split_percentage > 0 and test_split_percentage < 1, "test size in (0,1)"
    sizes = {
        "small": {"size": 52601, "positive": 27834, "negative": 24767},
        "large": {"size": 275565, "positive": 147778, "negative": 127787}
    }

    test_ids = []
    for idx, example in enumerate(examples):
        if idx % 10000 == 0:
            print("[split_pan_dataset_closed_set_v1] processed %d examples" % (idx))
        
    # determine Train/Test sizes
    same_author_examples = [ex for ex in examples if ex['same']]
    diff_author_examples = [ex for ex in examples if not ex['same']]
    random.shuffle(diff_author_examples)
    sa_size = len(same_author_examples)
    da_size = len(diff_author_examples)
    sa_test_size = int(test_split_percentage * sa_size)
    sa_train_size = sa_size - sa_test_size
    da_test_size = int(test_split_percentage * da_size)
    da_train_size = da_size - da_test_size

    # retrieve documents of all same-author (SA) pairs
    # sa_docs = {'author_id': [ids of SA pairs of this authors]}
    sa_docs = defaultdict(list)
    for example in same_author_examples:
        author_id = example['authors'][0] 
        sa_docs[author_id].append(example['id'])

    # first, populate SA test set
    print("[split_pan_dataset_closed_set_v1] Adding same-author (SA) pairs to the test set")
    sa_test_count = 0
    for author_id, pair_ids in sa_docs.items():
        author_docs_num = len(pair_ids)
        if author_docs_num >= 2:
            test_ids += pair_ids[:author_docs_num // 2]
            sa_test_count += author_docs_num // 2
        if sa_test_count >= sa_test_size:
            break
    
    # add DA pairs to test set
    da_test_count = 0
    print("[split_pan_dataset_closed_set_v1] Adding different-author (DA) pairs to the test set")
    for idx, example in enumerate(diff_author_examples):
        if idx % 10000 == 0:
            print("[split_pan_dataset_closed_set_v1] processed %d examples" % (idx))
        test_ids.append(example['id'])
        da_test_count += 1
        if da_test_count == da_test_size:
            break

    test_ids_map = {test_id:1 for test_id in test_ids}
    train_ids = [example['id'] for example in examples if example['id'] not in test_ids_map]
    train_ids_map = {train_id:1 for train_id in train_ids}

    # statistics
    train_stats = defaultdict(int)
    train_stats['size'] = len(train_ids)
    test_stats = defaultdict(int)
    test_stats['size'] = len(test_ids)
    for example in examples:
        same_author = example['same']
        same_fandom = example['fandoms'][0] == example['fandoms'][1]
        stats_dict = train_stats if example['id'] in train_ids_map else test_stats
        if same_author:
            if same_fandom:
                stats_dict['sa_sf'] += 1
            else:
                stats_dict['sa_df'] += 1
        else:
            if same_fandom:
                stats_dict['da_sf'] += 1
            else:
                stats_dict['da_df'] += 1
    
    for split_name, stats_dict in zip(["TRAIN", "TEST"], [train_stats, test_stats]):
        print("%s size: %d" % (split_name, stats_dict['size']))
        print("    Same author pairs: ", stats_dict['sa_sf'] + stats_dict['sa_df'])
        print("        Same fandom pairs: ", stats_dict['sa_sf'])
        print("        Different fandom pairs: ", stats_dict['sa_df'])
        print("    Different author pairs: ", stats_dict['da_sf'] + stats_dict['da_df'])
        print("        Same fandom pairs: ", stats_dict['da_sf'])
        print("        Different fandom pairs: ", stats_dict['da_df'])

    return (train_ids, test_ids)


def split_pan_dataset_closed_set_v2(examples: List[Dict], 
                                    test_split_percentage: float) -> (List, List):
    """
    Split PAN 2020 dataset in Train and Test under the closed-set assumption*. This requires that
    authors in Test set appear in Train as well. However, due to the large number of authors 
    in the different-author (DA) pairs, it is difficult to achieve this strictly. We try to 
    make sure that at least one of the authors (ai, aj) in DA Test pairs appears in DA Train 
    pairs or in same-author (SA) Train pairs.

    Args:
        examples: dataset samples read from a .jsonl file using read_jsonl_examples()
        test_split_percentage: size of the Test split as a percentage of the whole dataset
    
    Returns a list of unique pair ids for each dataset split
    """
    assert test_split_percentage > 0 and test_split_percentage < 1, "test size in (0,1)"
    sizes = {
        "small": {"size": 52601, "positive": 27834, "negative": 24767},
        "large": {"size": 275565, "positive": 147778, "negative": 127787}
    }

    # determine Train/Test sizes
    same_author_examples = [ex for ex in examples if ex['same']]
    diff_author_examples = [ex for ex in examples if not ex['same']]
    random.shuffle(diff_author_examples)
    sa_size = len(same_author_examples)
    da_size = len(diff_author_examples)
    sa_test_size = int(test_split_percentage * sa_size)
    sa_train_size = sa_size - sa_test_size
    da_test_size = int(test_split_percentage * da_size)
    da_train_size = da_size - da_test_size
    
    # add test ids of same-author pairs
    test_ids = []
   
    # retrieve author ids of same-author (SA) pairs
    sa_authors_ids = set()
    for example in same_author_examples:
        sa_authors_ids.add(example['authors'][0])
    
    # Algorithm
    # Create dictionary of frequencies of each author in DA pairs
    # freq = {'a1': 5, 'a2': 4, 'a3': 1, ..., etc}
    # Go through the DA pairs (ai, aj)
    #   if (ai, aj) appear in other DA pairs
    #       move (ai, aj) to test and
    #       decrease frequencies
    #   else if (ai, aj) appear in SA test pairs:
    #       move (ai, aj) to test
    #   else
    #       move (ai, aj) to train
    
    # create frequency of author ids in DA pairs    
    diff_author_freq = defaultdict(int)
    for example in diff_author_examples:
        fst_author_id = example['authors'][0]
        snd_author_id = example['authors'][1]
        diff_author_freq[fst_author_id] += 1
        diff_author_freq[snd_author_id] += 1

    # populate test set with DA pairs (ai, aj) such that at least one of the authors 
    # ai or aj in the test split appears in other DA train pairs or in SA pairs
    test_ids = []
    test_author_ids = set()
    da_sf = 0
    for example in diff_author_examples:
        fst_author_id = example['authors'][0] # a1
        snd_author_id = example['authors'][1] # a2
        same_fandom = example['fandoms'][0] == example['fandoms'][1]

        # check if a1 or a2 appear in other DA pairs
        fst_frequent = diff_author_freq[fst_author_id] >= 2
        snd_frequent = diff_author_freq[snd_author_id] >= 2
        if fst_frequent or snd_frequent:
            test_ids.append(example['id'])
            if same_fandom:
                da_sf += 1
            if fst_frequent:
                test_author_ids.add(fst_author_id)
                diff_author_freq[fst_author_id] -= 1
            if snd_frequent:
                test_author_ids.add(snd_author_id)
                diff_author_freq[snd_author_id] -= 1
        # check if a1 or a2 appear in SA pairs
        elif fst_author_id in sa_authors_ids or snd_author_id in sa_authors_ids:
            test_ids.append(example['id'])
            if same_fandom:
                da_sf += 1
            if fst_author_id in sa_authors_ids:
                test_author_ids.add(fst_author_id)
            if snd_author_id in sa_authors_ids:
                test_author_ids.add(snd_author_id)

    da_ids = [example['id'] for example in diff_author_examples]
    train_ids = [ex_id for ex_id in da_ids if ex_id not in test_ids]

    print("Number of different-author (DA) pairs: ", len(diff_author_examples))
    print("    Number of candidate DA test pairs: ", len(test_ids))
    print("         of which same fandom: ", da_sf)
    print("         of which diff fandom: ", len(test_ids) - da_sf)
    print("    Number of candidate DA train pairs: ", len(train_ids))

    # if too many DA test candidates, trim them
    if len(test_ids) > da_test_size:
        print("We only need %d DA test examples, trimming %d examples" % \
            (da_test_size, len(test_ids)-da_test_size))
        test_ids = test_ids[:da_test_size]
        # update author ids in test
        test_author_ids = set()
        for example in diff_author_examples:
            if example['id'] in test_ids:
                test_author_ids.add(example['authors'][0])
                test_author_ids.add(example['authors'][1])
    else:
        # if not enough DA test pairs, add further DA pairs
        da_test_count = len(test_ids)
        print("Not enough DA test examples %d/%d, adding other pairs" % \
            (da_test_count, da_test_size))
        for example in diff_author_examples:
            if example['id'] not in test_ids:
                test_ids.append(example['id'])
                test_author_ids.add(example['authors'][0])
                test_author_ids.add(example['authors'][1])
                da_test_count += 1
                if da_test_count == da_test_size:
                    break

    # retrieve documents of all same-author (SA) pairs
    # sa_docs = {'author_id': [ids of SA pairs of this authors]}
    sa_docs = defaultdict(list)
    for example in same_author_examples:
        author_id = example['authors'][0] 
        sa_docs[author_id].append(example['id'])

    # first, populate SA test set with authors belonging to DA test set
    print("Adding same-author (SA) pairs to the test set (authors from DA)")
    sa_test_count = 0
    for author_id, pair_ids in sa_docs.items():
        if author_id not in test_author_ids:
            continue

        author_docs_num = len(pair_ids)
        if author_docs_num >= 2:
            test_ids += pair_ids[:author_docs_num // 2]
            test_author_ids.add(author_id)
            sa_test_count += author_docs_num // 2
        if sa_test_count >= sa_test_size:
            break
    
    if sa_test_count >= sa_test_size:
        print("Added SA examples to test set: %d/%d" % (sa_test_count, sa_test_size))
    else:
        print("Not enough SA examples in test set: %d/%d, adding others" % (sa_test_count, sa_test_size))
        # if not enough, populate SA test set with other authors as well

        for author_id, pair_ids in sa_docs.items():
            # we have already added this author
            if author_id in test_author_ids:
                continue

            author_docs_num = len(pair_ids)
            if author_docs_num >= 2:
                test_ids += pair_ids[:author_docs_num // 2]
                test_author_ids.add(author_id)
                sa_test_count += author_docs_num // 2
            if sa_test_count >= sa_test_size:
                break
        print("Completed SA examples in test set: %d/%d" % (sa_test_count, sa_test_size))

    
    test_ids_map = {test_id:1 for test_id in test_ids}
    train_ids = [example['id'] for example in examples if example['id'] not in test_ids_map]
    train_ids_map = {train_id:1 for train_id in train_ids}

    # statistics
    train_stats = defaultdict(int)
    test_stats = defaultdict(int)
    for example in examples:
        same_author = example['same']
        same_fandom = example['fandoms'][0] == example['fandoms'][1]
        stats_dict = train_stats if example['id'] in train_ids_map else test_stats
        if same_author:
            if same_fandom:
                stats_dict['sa_sf'] += 1
            else:
                stats_dict['sa_df'] += 1
        else:
            if same_fandom:
                stats_dict['da_sf'] += 1
            else:
                stats_dict['da_df'] += 1
    
    for split_name, stats_dict in zip(["TRAIN", "TEST"], [train_stats, test_stats]):
        split_ids = train_ids if split_name == 'TRAIN' else test_ids
        print("%s size: %d" % (split_name, len(split_ids)))
        print("    Same author pairs: ", stats_dict['sa_sf'] + stats_dict['sa_df'])
        print("        Same fandom pairs: ", stats_dict['sa_sf'])
        print("        Different fandom pairs: ", stats_dict['sa_df'])
        print("    Different author pairs: ", stats_dict['da_sf'] + stats_dict['da_df'])
        print("        Same fandom pairs: ", stats_dict['da_sf'])
        print("        Different fandom pairs: ", stats_dict['da_df'])

    return (train_ids, test_ids)


def split_pan_dataset_open_set_unseen_authors(examples: List[Dict],
                                              test_split_percentage:float) -> (List, List):
    """
    Split dataset into train in test such that authors from SA train pairs do not 
    do not appear in SA test pairs.
    Args:
        examples: dataset samples read from a .jsonl file using read_jsonl_examples()
        test_split_percentage: size of the Test split as a percentage of the whole dataset
    
    Returns a list of unique pair ids for each dataset split
    """
    sa_examples = [ex for ex in examples if ex['same']]
    sa_size = len(sa_examples)
    sa_test_size = int(test_split_percentage * sa_size)
    sa_train_size = sa_size - sa_test_size

    da_examples = [ex for ex in examples if not ex['same']]
    da_size = len(da_examples)
    da_test_size = int(test_split_percentage * da_size)
    da_train_size = da_size - da_test_size

    # {'author_id': {"ids": [SA_pair_ids],
    #                "fandoms": {fandoms}}
    sa_authors_train, sa_authors_test = defaultdict(dict), defaultdict(dict)
    test_ids = []
    for idx, example in enumerate(sa_examples):
        a = example['authors'][0]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        if a in sa_authors_train:
            sa_authors_train[a]['ids'].append(example['id'])
            sa_authors_train[a]['fandoms'].add(f1)
            sa_authors_train[a]['fandoms'].add(f2)
        else:
            sa_authors_train[a]['ids'] = [example['id']]
            sa_authors_train[a]['fandoms'] = set([f1, f2])

    # split authors in train and test groups
    sa_test_count = 0
    sa_fandoms_train , sa_fandoms_test = set(), set()
    for author, author_info in sa_authors_train.items():
        # add author's examples to test
        test_ids.extend(author_info['ids'])
        # update fandoms and author test info
        sa_fandoms_test.update(author_info['fandoms'])
        sa_authors_test[author] = author_info   

        sa_test_count += len(author_info['ids'])
        if sa_test_count >= sa_test_size:
            break
    
    # remove SA test authors from SA train
    for author in sa_authors_test:
        if author in sa_authors_train:
            del sa_authors_train[author]

    # compute fandoms in SA train
    for author_info, author_info in sa_authors_train.items():
        sa_fandoms_train.update(author_info['fandoms'])

    print("[open_set_unseen_authors] populated %d/%d examples in SA test " \
          % (sa_test_count, sa_test_size))
    print("[open_set_unseen_authors] #authors in SA train = ", len(sa_authors_train))
    print("[open_set_unseen_authors] #authors in SA test = ", len(sa_authors_test))
    print("[open_set_unseen_authors] overlapping authors = ", \
          len(sa_authors_train.keys() & sa_authors_test.keys()))
    print("[open_set_unseen_authors] #fandoms in SA train = ", len(sa_fandoms_train))
    print("[open_set_unseen_authors] #fandoms in SA test = ", len(sa_fandoms_test))
    print("[open_set_unseen_authors] overlapping fandoms = ", \
          len(sa_fandoms_train & sa_fandoms_test))
    
    # add DA examples to test
    da_fandoms_train, da_fandoms_test = set(), set()
    da_authors_train, da_authors_test = set(), set()
    da_test_count = 0
    for idx, example in enumerate(da_examples):
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        if a1 in sa_authors_test or a2 in sa_authors_test:
            test_ids.append(example['id'])
            da_test_count += 1
            da_fandoms_test.update([f1, f2])
            da_authors_test.update([a1, a2])
            if da_test_count >= da_test_size:
                break
    
    # add DA examples to test set until completion
    for idx, example in enumerate(da_examples):
        if da_test_count >= da_test_size:
            break
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        test_ids.append(example['id'])
        da_test_count += 1
        da_fandoms_test.update([f1, f2])
        da_authors_test.update([a1, a2])
                  
    # compute DA train authors and fandoms stats
    test_ids_map = {test_id:1 for test_id in test_ids}
    for idx, example in enumerate(da_examples):
        if example['id'] not in test_ids_map:
            a1, a2 = example['authors'][0], example['authors'][1]
            f1, f2 = example['fandoms'][0], example['fandoms'][1]
            da_fandoms_train.update([f1, f2])
            da_authors_train.update([a1, a2])

    print("[open_set_unseen_authors] populated %d/%d examples in DA test " \
          % (da_test_count, da_test_size))
    print("[open_set_unseen_authors] #authors in DA train = ", len(da_authors_train))
    print("[open_set_unseen_authors] #authors in DA test = ", len(da_authors_test))
    print("[open_set_unseen_authors] overlapping authors DA test ^ DA train = ", \
          len(da_authors_train & da_authors_test))
    print("[open_set_unseen_authors] overlapping authors DA test ^ SA train = ", \
          len(sa_authors_train.keys() & da_authors_test))
    print("[open_set_unseen_authors] #fandoms in DA train = ", len(da_fandoms_train))
    print("[open_set_unseen_authors] #fandoms in DA test = ", len(da_fandoms_test))
    print("[open_set_unseen_authors] overlapping fandoms = ", \
          len(da_fandoms_train & da_fandoms_test))

    train_ids = [example['id'] for example in examples if example['id'] not in test_ids_map]
    # statistics
    train_stats = defaultdict(int)
    train_stats['size'] = len(train_ids)
    test_stats = defaultdict(int)
    test_stats['size'] = len(test_ids)
    for example in examples:
        same_author = example['same']
        same_fandom = example['fandoms'][0] == example['fandoms'][1]
        stats_dict = test_stats if example['id'] in test_ids_map else train_stats
        if same_author:
            if same_fandom:
                stats_dict['sa_sf'] += 1
            else:
                stats_dict['sa_df'] += 1
        else:
            if same_fandom:
                stats_dict['da_sf'] += 1
            else:
                stats_dict['da_df'] += 1
    
    for split_name, stats_dict in zip(["TRAIN", "TEST"], [train_stats, test_stats]):
        print("%s size: %d" % (split_name, stats_dict['size']))
        print("    Same author pairs: ", stats_dict['sa_sf'] + stats_dict['sa_df'])
        print("        Same fandom pairs: ", stats_dict['sa_sf'])
        print("        Different fandom pairs: ", stats_dict['sa_df'])
        print("    Different author pairs: ", stats_dict['da_sf'] + stats_dict['da_df'])
        print("        Same fandom pairs: ", stats_dict['da_sf'])
        print("        Different fandom pairs: ", stats_dict['da_df'])

    return (train_ids, test_ids) 


def extract_fandoms_sample(examples: List[Dict]) -> Dict:
    # {"hp": ['32432321', '12312312',....]}
    fandoms = defaultdict(list)
    for idx, example in enumerate(examples):
        if idx % 10000 == 0: 
            print("Processed %d examples " % (idx))
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        d1, d2 = example['pair'][0], example['pair'][1]
        ex_id = example['id']

        if len(fandoms[f1]) < 5:
            fandoms[f1].append(d1)
        if len(fandoms[f2]) < 5:
            fandoms[f2].append(d2)

    return fandoms

def split_pan_small_dataset_open_set_unseen_fandoms(examples: List[Dict], 
                                                    test_split_percentage:float) -> (List, List, List):
    """
    Split XS dataset into train/val/test such that fandoms from val/test do not appear in train. 
    Some authors in train mai appear in val/test. Similar to ```split_pan_dataset_open_set_unseen_fandoms```,
    but simpler algorithm due to XS PAN dataset having no DA-SF pairs.
    
    Algorithm:
        1. Let F be the fandoms of SA pairs (same-author pairs)
        2. Split F into two disjoint sets F_train and F_test
        2. populate test set with SA pairs whose fandoms are in F_test until enough examples
           *remove SA train pairs if either f1 or f2 appear in F_test
        3. populate test set with DA pairs (a1, a2, f1, f2) whose fandoms f1, f2 are both in F_test
           *remove DA train pairs if either f1 or f2 appear in F_test
    Args:
        examples: dataset samples read from a .jsonl file using read_jsonl_examples()
        test_split_percentage: size of the Test split as a percentage of the whole dataset
    
    Returns a list of unique pair ids for each dataset split
    """
    sa_examples = [ex for ex in examples if ex['same']]
    sa_size = len(sa_examples)
    sa_test_size = int(test_split_percentage * sa_size)
    sa_train_size = sa_size - sa_test_size

    da_examples = [ex for ex in examples if not ex['same']]
    da_size = len(da_examples)
    da_test_size = int(test_split_percentage * da_size)
    da_train_size = da_size - da_test_size

    # create SA fandoms
    fandoms_sa_train, fandoms_sa_test = defaultdict(dict), {}
    for idx, example in enumerate(sa_examples):
        if idx % 10000 == 0:
            print("[open_set_unseen_fandoms] processed %d SA examples" % (idx))
        ex_id = example['id']
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        assert a1 == a2, "Different authors"

        for a,f in zip([a1, a1], [f1, f2]):
            if f not in fandoms_sa_train:
                fandoms_sa_train[f]['ids'] = [ex_id]
                fandoms_sa_train[f]['authors'] = set([a])
            else:
                fandoms_sa_train[f]['ids'].append(ex_id)
                fandoms_sa_train[f]['authors'].add(a)
    
    # split SA fandoms in F_train and F_test
    # sort dictionary from least popular fandoms to most popular
    # move fandoms from fandoms_sa_train to fandoms_sa_test until enough SA test examples
    least_freq_fandoms_train = sorted(fandoms_sa_train.items(), key=lambda x: len(x[1]['ids']))
    #least_freq_fandoms_train = fandoms_sa_train.items()
    authors_sa_train, authors_sa_test = {}, {}
    sa_test_count = 0
    sa_test_ids = []
    for f, f_info in least_freq_fandoms_train:
        for a in f_info['authors']:
            authors_sa_test[a] = 1
        for pair_id in f_info['ids']:
            sa_test_ids.append(pair_id)
        # move fandom info to the fandom test group
        fandoms_sa_test[f] = f_info
        sa_test_count += len(f_info['ids'])
        if sa_test_count >= sa_test_size:
            break

    # remove SA test fandoms from SA train fandoms
    for fandom in fandoms_sa_test.keys():
        if fandom in fandoms_sa_train:
            del fandoms_sa_train[fandom]

    # pull authors from SA train fandoms
    for fandom, fandom_info in fandoms_sa_train.items():
        for a in fandom_info['authors']:
            authors_sa_train[a] = 1
    
    print("[open_set_unseen_fandoms] Populated %d out of %d SA test examples " \
            % (sa_test_count, sa_test_size))
    extra = sa_test_count - sa_test_size
    sa_train_size -= extra
    Sa_test_size = sa_test_count
    print("[open_set_unseen_fandoms] #fandoms in SA train group ", len(fandoms_sa_train))
    print("[open_set_unseen_fandoms] #fandoms in SA test group ", len(fandoms_sa_test))
    print("[open_set_unseen_fandoms] overlapping #fandoms SA train & SA test", \
            len(fandoms_sa_train.keys() & fandoms_sa_test.keys()))
    print("[open_set_unseen_fandoms] #authors in SA train group ", len(authors_sa_train))
    print("[open_set_unseen_fandoms] #authors in SA test group ", len(authors_sa_test))
    print("[open_set_unseen_fandoms] overlapping #authors SA test & SA train ", \
            len(authors_sa_train.keys() & authors_sa_test.keys()))
    print("[open_set_unseen_fandoms] =======================================================")

    # add DA examples to test set
    da_test_count = 0
    da_test_ids = []
    authors_da_train, authors_da_test = {}, {}
    fandoms_da_train, fandoms_da_test = {}, {}
    for idx, example in enumerate(da_examples):
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        # limit DA-DF examples to expected size
        if f1 in fandoms_sa_test and f2 in fandoms_sa_test:
            # adding example to DA-DF test pairs
            da_test_ids.append(example['id'])
            da_test_count += 1
            fandoms_da_test[f1], fandoms_da_test[f2] = 1, 1
            authors_da_test[a1], authors_da_test[a2] = 1, 1
            #if da_test_count == da_test_size:
            #    break

    # create DA-DF train fandoms and train authors stats
    test_ids_map = {test_id:1 for test_id in sa_test_ids+da_test_ids}
    dropped_da_train = 0
    dropped_train_ids = {}
    for idx, example in enumerate(da_examples):
        if example['id'] in test_ids_map:
            continue
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        if f1 in fandoms_sa_test or f2 in fandoms_sa_test:
            dropped_da_train += 1
            dropped_train_ids[example['id']] = 1
            continue
        fandoms_da_train[f1] = 1
        fandoms_da_train[f2] = 1
        authors_da_train[a1] = 1
        authors_da_train[a2] = 1

    print("[open_set_unseen_fandoms] Populated %d out of %d DA test examples " \
          % (da_test_count, da_test_size))
    print("[open_set_unseen_fandoms] dropped %d/%d in DA train group " % (dropped_da_train, da_train_size))
    print("[open_set_unseen_fandoms] #fandoms in DA train group ", len(fandoms_da_train))
    print("[open_set_unseen_fandoms] #fandoms in DA test group ", len(fandoms_da_test))
    print("[open_set_unseen_fandoms] overlapping #fandoms DA test & DA train", \
            len(fandoms_da_train.keys() & fandoms_da_test.keys()))
    print("[open_set_unseen_fandoms] overlapping #fandoms DA test & SA train", \
            len(fandoms_da_train.keys() & fandoms_da_test.keys()))
    print("[open_set_unseen_fandoms] #authors in DA train group ", len(authors_da_train))
    print("[open_set_unseen_fandoms] #authors in DA test group ", len(authors_da_test))
    print("[open_set_unseen_fandoms] overlapping #authors in DA test & DA train ", \
            len(authors_da_test.keys() & authors_da_train.keys()))
    print("[open_set_unseen_fandoms] overlapping #authors in DA test & SA train ", \
            len(authors_da_test.keys() & authors_sa_train.keys()))

    train_ids = []
    for example in examples:
        if example['id'] not in test_ids_map and example['id'] not in dropped_train_ids:
            train_ids.append(example['id'])
    train_ids_map = {train_id:1 for train_id in train_ids}

    # create val and test ids
    print("[open_set_unseen_fandoms] val+test = %d" % (len(sa_test_ids)+len(da_test_ids)))
    print("[open_set_unseen_fandoms]    SA val+test = %d" % (len(sa_test_ids)))
    print("[open_set_unseen_fandoms]    DA val+test = %d" % (len(da_test_ids)))
    val_ids = sa_test_ids[:len(sa_test_ids)//2]
    test_ids = sa_test_ids[len(sa_test_ids)//2:]
    val_ids += da_test_ids[:len(da_test_ids)//2]
    test_ids += da_test_ids[len(da_test_ids)//2:]
    print("[open_set_unseen_fandoms] val size = %d" % (len(val_ids)))
    print("[open_set_unseen_fandoms] test size = %d" % (len(test_ids)))
    
    val_ids_map = {val_id:1 for val_id in val_ids}
    test_ids_map = {test_id:1 for test_id in test_ids}

    # statistics
    train_stats = defaultdict(int)
    val_stats = defaultdict(int)
    test_stats = defaultdict(int)
    for example in examples:
        same_author = example['same']
        same_fandom = example['fandoms'][0] == example['fandoms'][1]
        if example['id'] in test_ids_map:
            stats_dict = test_stats
        elif example['id'] in val_ids_map:
            stats_dict = val_stats
        elif example['id'] in train_ids_map:
            stats_dict = train_stats
        else:
            continue
        
        if same_author:
            if same_fandom:
                stats_dict['sa_sf'] += 1
            else:
                stats_dict['sa_df'] += 1
        else:
            if same_fandom:
                stats_dict['da_sf'] += 1
            else:
                stats_dict['da_df'] += 1
    
    split_names = ['TRAIN', 'VAL', 'TEST']
    split_sizes = [len(train_ids), len(val_ids), len(test_ids)]
    split_stats = [train_stats, val_stats, test_stats]
    for split_name, split_size, stats_dict in zip(split_names, split_sizes, split_stats):
        print("%s size: %d" % (split_name, split_size))
        print("    Same author pairs: ", stats_dict['sa_sf'] + stats_dict['sa_df'])
        print("        Same fandom pairs: ", stats_dict['sa_sf'])
        print("        Different fandom pairs: ", stats_dict['sa_df'])
        print("    Different author pairs: ", stats_dict['da_sf'] + stats_dict['da_df'])
        print("        Same fandom pairs: ", stats_dict['da_sf'])
        print("        Different fandom pairs: ", stats_dict['da_df'])
    
    return (train_ids, val_ids, test_ids)

def split_pan_dataset_open_set_unseen_fandoms(examples: List[Dict], 
                                              test_split_percentage:float) -> (List, List, List):
    """
    Split XL dataset into train/val/test test such that fandoms from train
    do not appear in val/test. Authors in train mai appear in val/test.
    Algorithm:
        1. Let F be the fandoms of DA-SF pairs (different-author same fandom)
        2. Split F into two disjoint sets F_train and F_test
        2. populate test set with DA-SF pairs whose fandoms are in F_test until enough examples
        5. populate test set with SA pairs (a1, a1, f1, f2) whose fandoms f1, f2 are both in F_test
           *remove SA train pairs if either f1 or f2 appear in F_test
        6. populate test set with DA-DF pairs (a1, a2, f1, f2) whose fandoms f1, f2 are both in F_test
           *remove DA-DF train pairs if either f1 or f2 appear in F_test
        7. Split test set into val/test
    Args:
        examples: dataset samples read from a .jsonl file using read_jsonl_examples()
        test_split_percentage: size of the Val+Test split as a percentage of the whole dataset
                               if 0.1 then train/val/split percentages are 90%/5%/5%.
    
    Returns a list of unique pair ids for each train/val/test dataset split
    """
    sa_examples = [ex for ex in examples if ex['same']]
    random.shuffle(sa_examples)
    sa_size = len(sa_examples)
    sa_test_size = int(test_split_percentage * sa_size)
    sa_train_size = sa_size - sa_test_size

    da_examples = [ex for ex in examples if not ex['same']]
    da_sf_examples = [ex for ex in da_examples if ex['fandoms'][0] == ex['fandoms'][1]]
    da_df_examples = [ex for ex in da_examples if ex['fandoms'][0] != ex['fandoms'][1]]
    da_size = len(da_examples)
    da_sf_size = len(da_sf_examples)
    da_df_size = da_size - da_sf_size

    da_sf_test_size = int(test_split_percentage * da_sf_size)
    da_sf_train_size = da_sf_size - da_sf_test_size
    da_df_test_size = int(test_split_percentage * da_df_size)
    da_df_train_size = da_df_size - da_df_test_size

    da_test_size = da_sf_test_size + da_df_test_size
    da_train_size = da_sf_train_size + da_df_train_size

    test_ids = []
    val_ids = []
    sa_test_ids, da_sf_test_ids, da_df_test_ids = [], [], []
    # fandoms_train = {"fandom": {"ids": [$id, $id, ,,,],
    #                             "authors: [a1, a2, ...]}
    #                 }
    fandoms_da_sf_train, fandoms_da_sf_test = defaultdict(dict), {}
    authors_da_sf_train = {}
    for idx, example in enumerate(da_sf_examples):
        if idx % 10000 == 0:
            print("[open_set_unseen_fandoms] processed %d DA-SF examples" % (idx))
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        assert f1 == f2, "Different fandoms"
        
        if f1 not in fandoms_da_sf_train:
            fandoms_da_sf_train[f1]['ids'] = [example['id']]
            fandoms_da_sf_train[f1]['authors'] = set([a1, a2])
        else:
            fandoms_da_sf_train[f1]['ids'].append(example['id'])
            fandoms_da_sf_train[f1]['authors'].add(a1)
            fandoms_da_sf_train[f1]['authors'].add(a2)

    print("[open_set_unseen_fandoms] #fandoms in DA-SF = ", len(fandoms_da_sf_train))
    # populate test set with DA-SF examples
    da_sf_test_count = 0
    authors_da_sf_test = {}
    least_freq_fandoms_train = sorted(fandoms_da_sf_train.items(), key=lambda x: len(x[1]['ids']))
    for fandom, fandom_info in least_freq_fandoms_train:
        for a in fandom_info['authors']:
            authors_da_sf_test[a] = 1
        for pair_id in fandom_info['ids']:
            da_sf_test_ids.append(pair_id)
        
        # move fandom info to the fandom test group
        fandoms_da_sf_test[fandom] = fandom_info
        da_sf_test_count += len(fandom_info['ids'])
        if da_sf_test_count >= da_sf_test_size:
            break
    
    # remove DA-SF test fandoms from DA-SF train fandoms
    for fandom in fandoms_da_sf_test.keys():
        if fandom in fandoms_da_sf_train:
            del fandoms_da_sf_train[fandom]

    # pull authors from DA-SF train fandoms
    for fandom, fandom_info in fandoms_da_sf_train.items():
        for a in fandom_info['authors']:
            authors_da_sf_train[a] = 1
    
    print("[open_set_unseen_fandoms] Populated %d out of %d DA-SF test examples " \
            % (da_sf_test_count, da_sf_test_size))
    extra = da_sf_test_count - da_sf_test_size
    da_sf_train_size -= extra
    da_sf_test_size = da_sf_test_count

    print("[open_set_unseen_fandoms] #fandoms in DA-SF train group ", len(fandoms_da_sf_train))
    print("[open_set_unseen_fandoms] #fandoms in DA-SF test group ", len(fandoms_da_sf_test))
    print("[open_set_unseen_fandoms] overlapping fandoms ", \
            len(fandoms_da_sf_train.keys() & fandoms_da_sf_test.keys()))
    print("[open_set_unseen_fandoms] #authors in DA-SF train group ", len(authors_da_sf_train))
    print("[open_set_unseen_fandoms] #authors in DA-SF test group ", len(authors_da_sf_test))
    print("[open_set_unseen_fandoms] overlapping #authors ", \
            len(authors_da_sf_train.keys() & authors_da_sf_test.keys()))
    print("[open_set_unseen_fandoms] =======================================================")

    # add SA examples to test whose fandoms overlap with DA-SF test
    authors_sa_train, authors_sa_test = {}, {}
    fandoms_sa_train, fandoms_sa_test = {}, {}
    sa_test_count = 0
    for idx, example in enumerate(sa_examples):
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        if f1 in fandoms_da_sf_test and f2 in fandoms_da_sf_test:
            # adding example to SA test pairs
            sa_test_ids.append(example['id'])
            sa_test_count += 1
            # update fandoms and authors stats
            fandoms_sa_test[f1] = 1
            fandoms_sa_test[f2] = 1
            authors_sa_test[a1] = 1
            # if sa_test_count == sa_test_size:
            #     break

    # update counts
    print("[open_set_unseen_fandoms] Populated %d out of %d SA test examples " % (sa_test_count, sa_test_size))
    extra = sa_test_count - sa_test_size
    sa_train_size -= extra
    sa_test_size = sa_test_count
    test_size_so_far = sa_test_size + da_sf_test_size
    test_ids = sa_test_ids + da_sf_test_ids
    assert len(test_ids) == test_size_so_far, \
          "len(test_ids) = %d, test_size_so_far = %d" % (len(test_ids), test_size_so_far)
    
    test_ids_map = {test_id:1 for test_id in test_ids}
    # create author and fandom SA train stats
    dropped_sa_train = 0
    dropped_train_ids = {}
    for idx, example in enumerate(sa_examples):
        if example['id'] in test_ids_map:
            continue
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        if f1 in fandoms_da_sf_test or f2 in fandoms_da_sf_test:
            dropped_sa_train += 1
            dropped_train_ids[example['id']] = 1
            continue
        fandoms_sa_train[f1] = 1
        fandoms_sa_train[f2] = 1
        authors_sa_train[a1] = 1

    print("[open_set_unseen_fandoms] dropped %d/%d SA train examples " % (dropped_sa_train, sa_train_size))
    print("[open_set_unseen_fandoms] #fandoms in SA train group ", len(fandoms_sa_train))
    print("[open_set_unseen_fandoms] #fandoms in SA test group ", len(fandoms_sa_test))
    print("[open_set_unseen_fandoms] overlapping fandoms ", \
            len(fandoms_sa_train.keys() & fandoms_sa_test.keys()))
    print("[open_set_unseen_fandoms] #authors in SA train group ", len(authors_sa_train))
    print("[open_set_unseen_fandoms] #authors in SA test group ", len(authors_sa_test))
    print("[open_set_unseen_fandoms] overlapping #authors SA test & SA train ", \
            len(authors_sa_train.keys() & authors_sa_test.keys()))
    print("[open_set_unseen_fandoms] overlapping #authors SA test & DA-SF train ", \
            len(authors_da_sf_train.keys() & authors_sa_test.keys()))

    print("[open_set_unseen_fandoms] =======================================================")
    print("[open_set_unseen_fandoms] adding DA-DF pairs to test set")
    da_test_count = 0
    fandoms_da_df_train, fandoms_da_df_test = {}, {}
    authors_da_df_train, authors_da_df_test = {}, {}

    # add DA-DF examples to test set
    da_df_test_count = 0
    for idx, example in enumerate(da_df_examples):
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        # limit DA-DF examples to expected size
        if f1 in fandoms_da_sf_test and f2 in fandoms_da_sf_test:
            # adding example to DA-DF test pairs
            da_df_test_ids.append(example['id'])
            da_df_test_count += 1
            fandoms_da_df_test[f1], fandoms_da_df_test[f2] = 1, 1
            authors_da_df_test[a1], authors_da_df_test[a2] = 1, 1
            if da_df_test_count == da_df_test_size:
                break

    # create DA-DF train fandoms and train authors stats
    test_ids = test_ids + da_df_test_ids
    test_ids_map = {test_id:1 for test_id in test_ids}
    dropped_da_df_train = 0
    for idx, example in enumerate(da_df_examples):
        if example['id'] in test_ids_map:
            continue
        a1, a2 = example['authors'][0], example['authors'][1]
        f1, f2 = example['fandoms'][0], example['fandoms'][1]
        if f1 in fandoms_da_sf_test or f2 in fandoms_da_sf_test:
            dropped_da_df_train += 1
            dropped_train_ids[example['id']] = 1
            continue
        fandoms_da_df_train[f1] = 1
        fandoms_da_df_train[f2] = 1
        authors_da_df_train[a1] = 1

    print("[open_set_unseen_fandoms] Populated %d out of %d DA-DF test examples " \
          % (da_df_test_count, da_df_test_size))
    print("[open_set_unseen_fandoms] dropped %d/%d in DA-DF train group " % (dropped_da_df_train, da_df_train_size))
    print("[open_set_unseen_fandoms] #fandoms in DA-DF train group ", len(fandoms_da_df_train))
    print("[open_set_unseen_fandoms] #fandoms in DA-DF test group ", len(fandoms_da_df_test))
    print("[open_set_unseen_fandoms] overlapping fandoms ", \
            len(fandoms_da_df_train.keys() & fandoms_da_df_test.keys()))
    print("[open_set_unseen_fandoms] #authors in DA-DF train group ", len(authors_da_df_train))
    print("[open_set_unseen_fandoms] #authors in DA-DF test group ", len(authors_da_df_test))
    print("[open_set_unseen_fandoms] overlapping #authors in DA-DF test & DA-DF train ", \
            len(authors_da_df_train.keys() & authors_da_df_test.keys()))
    print("[open_set_unseen_fandoms] overlapping #authors in DA-DF test & SA train ", \
            len(authors_sa_train.keys() & authors_da_df_test.keys()))
    print("[open_set_unseen_fandoms] overlapping #authors in DA-DF test & DA-SF train ", \
            len(authors_da_sf_train.keys() & authors_da_df_test.keys()))
      

    # create train ids
    train_ids = []
    for example in examples:
        if example['id'] not in test_ids_map and example['id'] not in dropped_train_ids:
            train_ids.append(example['id'])
    train_ids_map = {train_id:1 for train_id in train_ids}

    # create val and test ids
    print("[open_set_unseen_fandoms] val+test = %d" % (len(test_ids)))
    print("[open_set_unseen_fandoms]    DA-SF val+test = %d" % (len(da_sf_test_ids)))
    print("[open_set_unseen_fandoms]    SA val+test = %d" % (len(sa_test_ids)))
    print("[open_set_unseen_fandoms]    DA-DF val+test = %d" % (len(da_df_test_ids)))
    val_ids += da_sf_test_ids[:len(da_sf_test_ids)//2]
    test_ids = da_sf_test_ids[len(da_sf_test_ids)//2:]
    val_ids += sa_test_ids[:len(sa_test_ids)//2]
    test_ids += sa_test_ids[len(sa_test_ids)//2:]
    val_ids += da_df_test_ids[:len(da_df_test_ids)//2]
    test_ids += da_df_test_ids[len(da_df_test_ids)//2:]
    print("[open_set_unseen_fandoms] val size = %d" % (len(val_ids)))
    print("[open_set_unseen_fandoms] test size = %d" % (len(test_ids)))
    
    val_ids_map = {val_id:1 for val_id in val_ids}
    test_ids_map = {test_id:1 for test_id in test_ids}

    # statistics
    train_stats = defaultdict(int)
    val_stats = defaultdict(int)
    test_stats = defaultdict(int)
    for example in examples:
        same_author = example['same']
        same_fandom = example['fandoms'][0] == example['fandoms'][1]
        if example['id'] in test_ids_map:
            stats_dict = test_stats
        elif example['id'] in val_ids_map:
            stats_dict = val_stats
        elif example['id'] in train_ids_map:
            stats_dict = train_stats
        else:
            continue
        
        if same_author:
            if same_fandom:
                stats_dict['sa_sf'] += 1
            else:
                stats_dict['sa_df'] += 1
        else:
            if same_fandom:
                stats_dict['da_sf'] += 1
            else:
                stats_dict['da_df'] += 1
    
    split_names = ['TRAIN', 'VAL', 'TEST']
    split_sizes = [len(train_ids), len(val_ids), len(test_ids)]
    split_stats = [train_stats, val_stats, test_stats]
    for split_name, split_size, stats_dict in zip(split_names, split_sizes, split_stats):
        print("%s size: %d" % (split_name, split_size))
        print("    Same author pairs: ", stats_dict['sa_sf'] + stats_dict['sa_df'])
        print("        Same fandom pairs: ", stats_dict['sa_sf'])
        print("        Different fandom pairs: ", stats_dict['sa_df'])
        print("    Different author pairs: ", stats_dict['da_sf'] + stats_dict['da_df'])
        print("        Same fandom pairs: ", stats_dict['da_sf'])
        print("        Different fandom pairs: ", stats_dict['da_df'])

    return (train_ids, val_ids, test_ids)
        

def make_two_author_groups(authors_source: Union[str, Dict]) -> (Dict, Dict):
    """
    Algorithm 1 from Boenninghoff et al. 2020
    Splits a list of authors and their documents into 2 groups:
     - a group of authors with only 1 document
     - a group of authors with 2 ore more documents (even number)
    Arguments:
        authors_source: either a ```str``` indicating the path to an author folder or a ```Dict```
                        with author data read from the author folder. An author folder
                        contains .jsonl files, one per each author and is created via 
                        ```get_author_docs_from_jsonl```. The author data is a dictionary 
                        read from the abovementioned folder using ```get_author_data_from_folder```
    """
    single_doc_authors = {}
    even_doc_authors = {}
    if type(authors_source) == 'str':
        for idx, author_file in enumerate(os.listdir(authors_source)):
            if idx % 1000 == 0:
                print("[make_two_author_groups] processed ", idx, " authors")
            author_id = author_file[:-6]
            with open(os.path.join(authors_source, author_file)) as fp:
                author_data = []
                for line in fp:
                    author_data.append(json.loads(line))
                
                assert len(author_data) > 0, "error in loading %s" % author_file
                
                if len(author_data) == 1:
                    single_doc_authors[author_id] = author_data
                elif len(author_data) % 2 == 0:
                    even_doc_authors[author_id] = author_data
                else:
                    single_doc_authors[author_id] = [author_data[0]]
                    even_doc_authors[author_id] = author_data[1:]
    elif type(authors_source) == defaultdict or type(authors_source) == Dict:
        for author_id, author_data in authors_source.items():
            if len(author_data) == 1:
                single_doc_authors[author_id] = author_data
            elif len(author_data) % 2 == 0:
                even_doc_authors[author_id] = author_data
            else:
                single_doc_authors[author_id] = [author_data[0]]
                even_doc_authors[author_id] = author_data[1:]

    for author, docs in single_doc_authors.items():
        assert len(docs) == 1, "single-doc author group has more than one document"

    for author, docs in even_doc_authors.items():
        assert len(docs) > 0 and len(docs) % 2 == 0, "even-doc author group has odd number of documents" 
    
    return single_doc_authors, even_doc_authors


def clean_after_sampling(author_id: str, 
                         author_docs: List[Dict], 
                         single_doc_authors: Dict[str, List], 
                         even_doc_authors: Dict[str, List]):
    """
    Algorithm 2 from from Boenninghoff et al. 2020

    Args:
        author_id (str): [description]
        author_docs (List[Dict]): [description]
        single_doc_authors (Dict[List]): [description]
        even_doc_authors (Dict[List]): [description]

    Returns:
        a new example (optional) and the two updated author groups
    """
    new_example = None
    if len(author_docs) > 1:
        even_doc_authors[author_id] = author_docs
    elif len(author_docs) == 1:
        doc = author_docs[0]
        f1 = doc['fandom']
        d1 = doc['text']
        # check if author in single-doc author group
        if author_id in single_doc_authors:
            doc = single_doc_authors[author_id][0]
            f2 = doc['fandom']
            d2 = doc['text']
            new_example = {
                "same": True,
                "authors": [author_id, author_id],
                "fandoms": [f1, f2],
                #"pair": [d1, d2],
            }
            del single_doc_authors[author_id]
        else:
            single_doc_authors[author_id] = author_docs

    return new_example, single_doc_authors, even_doc_authors


def sample_pairs(authors_data: Dict, output_folder: str):
    """
    Algorithm 3 from Boenninghoff et al. 2020
    Returns same-author pairs as well as different-author pairs from the given
    authors data.

    Args:
        authors_data (Dict): authors data
        output_folder (str): [description]
    """
    if not os.path.exists(output_folder):
        print("[sample_pairs] folder %s doesn't exist, trying to create it" % (output_folder))
        os.mkdir(output_folder)

    single_doc_authors, even_doc_authors = make_two_author_groups(authors_data)
    samples_count = 0
    while len(single_doc_authors) > 1 and len(even_doc_authors) > 0:
        if samples_count % 100 == 0:
            print("[sample_pairs] samples_count = ", samples_count)
        # sample same-author pair
        if len(even_doc_authors) > 0:
            # sample author
            author_id = random.choice(list(even_doc_authors.keys()))
            author_docs = even_doc_authors[author_id]
            del even_doc_authors[author_id]

            # sample two documents from author's documents
            same_docs = random.sample(author_docs, k=2)
            f1, f2 = same_docs[0]['fandom'], same_docs[1]['fandom']
            d1, d2 = same_docs[1]['text'], same_docs[1]['text']

            # create same-author pair
            example = {
                "same": True, "authors": [author_id, author_id], "fandoms": [f1, f2]#, 
                #"pair": [d1, d2]
            }
            with open(os.path.join(output_folder, str(samples_count) + ".json"), "w") as fp:
                json.dump(example, fp)
                fp.write('\n')
                samples_count += 1

            # remove sampled documents
            author_docs = [doc for doc in author_docs if doc not in same_docs]

            # add remaining docs to even-docs author group
            example, single_doc_authors, even_doc_authors = clean_after_sampling(
                author_id, author_docs, single_doc_authors, even_doc_authors
            )
            if example:
                with open(os.path.join(output_folder, str(samples_count) + ".json"), "w") as fp:
                    json.dump(example, fp)
                    fp.write('\n')
                    samples_count += 1

        # sample different-author pair
        if len(even_doc_authors) > 1:
            # sample two authors
            author_ids = random.sample(list(even_doc_authors.keys()), k=2)
            fst_author_id, snd_author_id = author_ids[0], author_ids[1]
            fst_docs, snd_docs = even_doc_authors[fst_author_id], even_doc_authors[snd_author_id]
            
            # remove authors from group
            del even_doc_authors[fst_author_id]
            del even_doc_authors[snd_author_id]
            
            fst_fandoms = set([doc['fandom'] for doc in fst_docs])
            snd_fandoms = set([doc['fandom'] for doc in snd_docs])
            common_fandoms = list(fst_fandoms.intersection(snd_fandoms))
            #print("Common fandoms = ", common_fandoms)

            if len(common_fandoms) > 0:
                # try to sample same-fandom pair
                f = random.choice(common_fandoms)
                fst_docs_population = [doc for doc in fst_docs if doc['fandom'] == f]
                snd_docs_population = [doc for doc in snd_docs if doc['fandom'] == f]
            else:
                fst_docs_population = fst_docs
                snd_docs_population = snd_docs

            # take random doc from 1st author and random doc from 2nd author
            fst_doc = random.choice(fst_docs_population)
            snd_doc = random.choice(snd_docs_population)

            # remove documents from each author
            fst_docs = [doc for doc in fst_docs if doc != fst_doc]
            snd_docs = [doc for doc in snd_docs if doc != snd_doc]

            # create example
            example = {
                "same": False,
                "authors": [fst_author_id, snd_author_id],
                "fandoms": [fst_doc["fandom"], snd_doc["fandom"]]#,
                #"pair": [fst_doc["text"], snd_doc["text"]]
            }
            with open(os.path.join(output_folder, str(samples_count) + ".json"), "w") as fp:
                json.dump(example, fp)
                fp.write('\n')
                samples_count += 1
            
            # add authors back to their groups
            for (author_id, author_docs) in zip(author_ids, [fst_docs, snd_docs]):
                example, single_doc_authors, even_doc_authors = clean_after_sampling(
                    author_id, author_docs, single_doc_authors, even_doc_authors
                )
                if example:
                    with open(os.path.join(output_folder, str(samples_count) + ".json"), "w") as fp:
                        json.dump(example, fp)
                        fp.write('\n')
                        samples_count += 1
        elif len(single_doc_authors) > 1:
            # sample two authors
            author_ids = random.sample(list(single_doc_authors.keys()), k=2)
            fst_author_id, snd_author_id = author_ids[0], author_ids[1]

            fst_doc, snd_doc = single_doc_authors[fst_author_id][0], single_doc_authors[snd_author_id][0]
            del single_doc_authors[fst_author_id]
            del single_doc_authors[snd_author_id]

            example = {
                "same": False,
                "authors": [fst_author_id, snd_author_id],
                "fandoms": [fst_doc["fandom"], snd_doc["fandom"]]#,
                #"pair": [fst_doc["text"], snd_doc["text"]]
            }
            with open(os.path.join(output_folder, str(samples_count) + ".json"), "w") as fp:
                json.dump(example, fp)
                fp.write('\n')
                samples_count += 1


def split_jsonl_dataset_resampling(path_to_original_jsonl: str,
                                   path_to_authors_json: str,
                                   path_to_train_jsonl: str, 
                                   path_to_val_jsonl: str, 
                                   path_to_test_jsonl: str, 
                                   train_size: int,
                                   test_size: int):
    """
    Split the PAN dataset into train/val/test splits by resampling possible new pairs unseen
    in PAN. 
    Args:
        path_to_original_jsonl: path to existing .jsonl file, such as ```pan20-av-large-no-text.jsonl```
        path_to_authors_json (str): path to .json file containing author->documents dictionary;
                                    if given, ```path_to_original_jsonl``` is ignored
        path_to_train_jsonl (str): path to .jsonl file where the training examples will be saved
        path_to_val_jsonl (str): path to .jsonl file where the validation examples will be saved
        path_to_test_jsonl (str): path to .jsonl file where the test examples will be saved
        train_size: size of the train split as number of examples
        test_size: size of the val/test split as number of examples
    """
    def create_example(label, a1, a2, f1, f2, d1, d2):
        return {
            'id': str(uuid.uuid1()),
            'same': label,
            'authors': [a1, a2],
            'fandoms': [f1, f2],
            'pair': [d1, d2]
        }
    
    def get_fandoms_from_author_docs(author_docs):
        f_docs = defaultdict(list)
        for e in author_docs:
            fandom, text = e['fandom'], e['text']
            f_docs[fandom].append(text)
        return f_docs

    def get_af_dicts(authors_to_docs):
        authors_to_fandoms = {}
        fandoms_to_authors = defaultdict(set)
        for a, docs in authors_to_docs.items():
            f_set = set([e['fandom'] for e in docs])
            authors_to_fandoms[a] = f_set
            for f in f_set:
                fandoms_to_authors[f].add(a)
        return authors_to_fandoms, fandoms_to_authors

    def save_examples_to_jsonl(jsonl_path, examples):
        print("Writing examples to %s" % (jsonl_path))
        with open(jsonl_path, "w") as f:
            for idx, example in enumerate(examples):
                if idx % 10000 == 0:
                    print("[split_jsonl_dataset] Wrote %d examples" % (idx))
                json.dump(example, f)
                f.write('\n')

    if path_to_original_jsonl:
        # gather examples into {author:documents} dictionary
        authors_to_docs = defaultdict(list)
        with open(path_to_original_jsonl) as fp:
            for idx, line in enumerate(fp):
                if idx % 1000 == 0:
                    print(idx)
                entry = json.loads(line)

                a1, a2 = entry['authors'][0], entry['authors'][1]
                d1, d2 = entry['pair'][0], entry['pair'][1]
                f1, f2 = entry['fandoms'][0], entry['fandoms'][1]
                for (a,f,d) in zip([a1, a2], [f1,f2], [d1, d2]):
                    duplicate = False
                    for entry in authors_to_docs[a]:
                        if entry['text'] == d:
                            duplicate = True
                            break
                    if not duplicate:
                        authors_to_docs[a].append({'fandom': f, 'text': d})
        
        if path_to_authors_json:
            json.dump(
                authors_to_docs, 
                open(path_to_authors_json, 'w'),
                indent=2
            )
    elif path_to_authors_json:
        print("[split_jsonl_dataset_resampling] loading authors")
        authors_to_docs = json.load(open(path_to_authors_json))
    else:
        return None
        
    authors_to_fandoms, fandoms_to_authors = get_af_dicts(authors_to_docs)
    all_fandoms = set(list(fandoms_to_authors.keys()))
    print("All fandoms = ", len(all_fandoms))
    
    most_diverse_authors = sorted(
        authors_to_fandoms.items(), 
        key=lambda x: len(x[1]),
        reverse=True
    )
    most_diverse_authors = [e[0] for e in most_diverse_authors]
    
    test_fandoms = set()
    test_authors = set()
    test_set_pairs = []
    # populate SA test set pairs
    print("[split_jsonl_dataset_resampling] Populating SA test pairs")
    while len(test_set_pairs) < test_size//2:
        for author_id in most_diverse_authors:
            author_docs = authors_to_docs[author_id]
            f_set = authors_to_fandoms[author_id]
            if len(f_set) > 1:
                break
            #author_id = random.choice(list(authors_to_docs.keys()))
            #f_set = set([e['fandom'] for e in author_docs])
            #if len(test_fandoms) == 0:
            #    break
            # select authors whose fandoms overlap with the previously selected ones
            # this way we ensure test fandoms are more 'clustered'
            #if not (fset&test_fandoms) or len(f_set) == 1:
            #    continue
        test_authors.add(author_id)
        most_diverse_authors.remove(author_id)
        
        # create SA pairs from distinct fandoms of this author 
        # f_docs = {'f1':[d1, d2], 'f2': [d3]}
        f_docs = get_fandoms_from_author_docs(author_docs)
        fset = set()
        while len(f_docs) >= 2:
            valid_fandoms = list(f_docs.keys())
            # choose 2 fandoms randomly
            f1, f2 = random.sample(valid_fandoms, 2)
            fset|=set([f1, f2])
            d1 = f_docs[f1].pop()
            d2 = f_docs[f2].pop()
            test_set_pairs.append(create_example(
                True, author_id, author_id, f1, f2, d1, d2
            ))
            if len(f_docs[f1]) == 0:
                del f_docs[f1]
            if len(f_docs[f2]) == 0:
                del f_docs[f2]
        test_fandoms|=f_set
    print("  test fandoms after adding test SA", len(test_fandoms))

    # populate DA test set pairs
    print("[split_jsonl_dataset_resampling]  Populating same-fandom DA test pairs")
    num_da_test_pairs = 0
    da_sf_size = test_size//4
    #while num_da_test_pairs <= da_sf_size:
    # create same-fandom DA pairs
    for f in test_fandoms:
        authors_f = fandoms_to_authors[f]&test_authors
        if len(authors_f) == 0:
            continue
        author_pairs = [(ai,aj) for ai in authors_f for aj in authors_f if ai != aj][:4]
        #print("  fandom = %s, number of authors_pairs = %d" % (f, len(author_pairs)))
        for (a1, a2) in author_pairs:
            d1 = next(e['text'] for e in authors_to_docs[a1] if e['fandom'] == f)
            d2 = next(e['text'] for e in authors_to_docs[a2] if e['fandom'] == f)
            test_set_pairs.append(create_example(
                False, a1, a2, f, f, d1, d2
            ))
            num_da_test_pairs += 1
        if num_da_test_pairs > da_sf_size:
            break
    print("   %d same-fandom DA test pairs created" % (num_da_test_pairs))
    print("[split_jsonl_dataset_resampling]  Populating cross-fandom DA test pairs")
    while num_da_test_pairs <= test_size//2:
        # create cross-fandom fairs
        a1, a2 = random.sample(list(authors_to_docs.keys()), 2)
        a1_docs, a2_docs = authors_to_docs[a1], authors_to_docs[a2]
        a1_entry = random.choice(a1_docs)
        a2_entry = random.choice(a2_docs)
        d1, d2 = a1_entry['text'], a2_entry['text']
        f1, f2 = a1_entry['fandom'], a2_entry['fandom']
        if f1 not in test_fandoms or f2 not in test_fandoms:
            continue
        test_set_pairs.append(create_example(
            False, a1, a2, f1, f2, d1, d2
        ))
        test_fandoms|=set([f1, f2])
        test_authors|=set([a1, a2])
        num_da_test_pairs += 1
    print("  completed cross-fandom DA examples, %d pairs in total" % (num_da_test_pairs)) 
    print("  test fandoms after test SA", len(test_fandoms))
    # delete test authors from authors_to_docs
    print("#authors before test pairs: ", len(authors_to_docs))
    for a in test_authors:
        authors_to_docs.pop(a, None)
    print("Removing documents whose fandoms are in test")
    for a, docs in list(authors_to_docs.items()):
        remaining_docs =  [doc for doc in docs if doc['fandom'] not in test_fandoms]
        if remaining_docs:
            authors_to_docs[a] = remaining_docs
        else:
            del authors_to_docs[a]
    print("#authors after removing documents in test fandoms: ", len(authors_to_docs))
    
    authors_to_fandoms, fandoms_to_authors = get_af_dicts(authors_to_docs)
    print("#authors after test pairs: ", len(authors_to_docs))


    # populate val set pairs
    val_fandoms = set()
    val_authors = set()
    val_set_pairs = []
    # populate SA val set pairs
    print("[split_jsonl_dataset_resampling] Populating SA val pairs")
    while len(val_set_pairs) < test_size//2:
        if len(val_set_pairs) % 1000 == 0:
            print("[split_jsonl_dataset_resampling]   %d pairs completed" % (len(val_set_pairs)))
        while True:
            author_id = random.choice(list(authors_to_docs.keys()))
            #author_docs = authors_to_docs[author_id]
            author_docs = [d for d in authors_to_docs[author_id] if d['fandom'] not in test_fandoms]
            f_set = set([e['fandom'] for e in author_docs])
            #f_set = authors_to_fandoms[author_id]
            if len(f_set) > 1:
                break
            # obsolete: eliminate documents of author a whose fandoms appear in test set
            #seen_authors = author_id in test_authors
            #seen_fandoms = bool(f_set&test_fandoms)
            #if len(f_set) > 1 and not seen_fandoms and not seen_authors:
            #    break
        val_authors.add(author_id)
        
        # create SA val pairs from distinct fandoms of this author 
        f_docs = get_fandoms_from_author_docs(author_docs)
        count = 0
        fset = set()
        while len(f_docs) >= 2:
            valid_fandoms = list(f_docs.keys())
            # choose 2 fandoms randomly
            f1, f2 = random.sample(valid_fandoms, 2)
            fset|=set([f1, f2])
            d1 = f_docs[f1].pop()
            d2 = f_docs[f2].pop()
            val_set_pairs.append(create_example(
                True, author_id, author_id, f1, f2, d1, d2
            ))
            count += 1
            if len(f_docs[f1]) == 0:
                del f_docs[f1]
            if len(f_docs[f2]) == 0:
                del f_docs[f2]
        val_fandoms|=f_set
        #print("  %d pairs added from author %s" % (count, author_id))
        #del authors_to_docs[author_id]

    # populate DA val set pairs
    print("[split_jsonl_dataset_resampling]  Populating same-fandom DA val pairs")
    num_da_val_pairs = 0
    
    # create same-fandom DA val pairs
    da_sf_size = test_size//4
    #while num_da_val_pairs <= da_sf_size:
    for f in val_fandoms:
        authors_f = fandoms_to_authors[f]-test_authors
        author_pairs = [(ai,aj) for ai in authors_f for aj in authors_f if ai != aj][:4]
        #print("  fandom = %s, number of authors_pairs = %d" % (f, len(author_pairs)))
        for (a1, a2) in author_pairs:
            d1 = next(e['text'] for e in authors_to_docs[a1] if e['fandom'] == f)
            d2 = next(e['text'] for e in authors_to_docs[a2] if e['fandom'] == f)
            val_set_pairs.append(create_example(
                False, a1, a2, f, f, d1, d2
            ))
            num_da_val_pairs += 1
        if num_da_val_pairs > da_sf_size:
            break

    print("[split_jsonl_dataset_resampling]  Populating cross-fandom DA val pairs")
    da_sf_size == test_size//4
    while num_da_val_pairs <= test_size//2:
        # sample 2 authors
        a1, a2 = random.sample(list(authors_to_docs.keys()), 2)
        a1_docs, a2_docs = authors_to_docs[a1], authors_to_docs[a2]
        a1_entry = random.choice(a1_docs)
        a2_entry = random.choice(a2_docs)
        d1, d2 = a1_entry['text'], a2_entry['text']
        f1, f2 = a1_entry['fandom'], a2_entry['fandom']
        if f1 not in val_fandoms or f2 not in val_fandoms:
            continue
        #seen_authors = a1 in test_authors or a2 in test_authors
        #seen_fandoms = f1 in test_fandoms or f2 in test_fandoms
        #if seen_authors or seen_fandoms:
        #    continue
        val_set_pairs.append(create_example(
            False, a1, a2, f1, f2, d1, d2
        ))
        val_fandoms|=set([f1, f2])
        val_authors|=set([a1, a2])
        num_da_val_pairs += 1

    # delete val authors from authors_to_docs
    for a in val_authors:
        authors_to_docs.pop(a, None)

    # for a, docs in list(authors_to_docs.items()):
    #     remaining_docs =  [doc for doc in docs if doc['fandom'] not in test_fandoms]
    #     if remaining_docs:
    #         authors_to_docs[a] = remaining_docs
    #     else:
    #         del authors_to_docs[a]
    authors_to_fandoms, fandoms_to_authors = get_af_dicts(authors_to_docs)
    print("#authors available after val pairs: ", len(authors_to_docs))
    print("  val fandoms", len(val_fandoms))
    print("  val_fandoms ^ test_fandoms = ", len(val_fandoms&test_fandoms))
    print("  #authors in test set: ", len(test_authors))
    print("  #authors in val set: ", len(val_authors))
    print("  #authors val^test set: ", len(val_authors&test_authors))

    # populate train test set pairs
    train_fandoms = set()
    train_authors = set()
    train_set_pairs = []
    # populate SA train set pairs
    print("[split_jsonl_dataset_resampling] Populating SA train pairs")
    while len(train_set_pairs) < train_size//2:
        if len(train_set_pairs) % 1000 == 0:
            print("[split_jsonl_dataset_resampling] %d SA train pairs added" % (len(train_set_pairs)))

        while True:
            author_id = random.choice(list(authors_to_docs.keys()))
            #obsolete: eliminate documents of author a whose fandoms appear in test set
            author_docs = [d for d in authors_to_docs[author_id] if d['fandom'] not in test_fandoms]
            #author_docs = authors_to_docs[author_id]
            
            #f_set = authors_to_fandoms[author_id]
            f_set = set([e['fandom'] for e in author_docs])
            #seen_authors = author_id in test_authors or author_id in val_authors
            #seen_fandoms = bool(f_set&test_fandoms) #or bool(f_set&val_fandoms)
            if len(f_set) > 1:# and not seen_fandoms and not seen_authors:
                break
        train_authors.add(author_id)
        train_fandoms|=f_set

        # create SA pairs from distinct fandoms of this author
        f_docs = get_fandoms_from_author_docs(author_docs)
        count = 0
        while len(f_docs) >= 2:
            valid_fandoms = list(f_docs.keys())
            # choose 2 fandoms randomly
            f1, f2 = random.sample(valid_fandoms, 2)
            d1 = f_docs[f1].pop()
            d2 = f_docs[f2].pop()
            train_set_pairs.append(create_example(
                True, author_id, author_id, f1, f2, d1, d2
            ))
            count += 1
            if len(f_docs[f1]) == 0:
                del f_docs[f1]
            if len(f_docs[f2]) == 0:
                del f_docs[f2]
        #del authors_to_docs[author_id]
    
    # populate DA same-fandom train set pairs
    print("[split_jsonl_dataset_resampling]  Populating same-fandom DA train pairs")
    num_da_train_pairs = 0
    da_sf_size = train_size//4
    for f in train_fandoms:
        authors_f = fandoms_to_authors[f]-test_authors
        if len(authors_f) == 0:
            continue
        author_pairs = [(ai,aj) for ai in authors_f for aj in authors_f if ai != aj][:160]
        #print("  fandom = %s, number of authors_pairs = %d" % (f, len(author_pairs)))
        for (a1, a2) in author_pairs:
            d1 = next(e['text'] for e in authors_to_docs[a1] if e['fandom'] == f)
            d2 = next(e['text'] for e in authors_to_docs[a2] if e['fandom'] == f)
            train_set_pairs.append(create_example(
                False, a1, a2, f, f, d1, d2
            ))
            num_da_train_pairs += 1
        if num_da_train_pairs > da_sf_size:
            break
    
    print("[split_jsonl_dataset_resampling]  Populating cross-fandom DA train pairs")
    while num_da_train_pairs <= train_size//2: 
        # sample 2 authors
        a1, a2 = random.sample(list(authors_to_docs.keys()), 2)
        a1_docs, a2_docs = authors_to_docs[a1], authors_to_docs[a2]
        a1_entry = random.choice(a1_docs)
        a2_entry = random.choice(a2_docs)
        d1, d2 = a1_entry['text'], a2_entry['text']
        f1, f2 = a1_entry['fandom'], a2_entry['fandom']
        #seen_authors = (a1 in test_authors or a2 in test_authors or
        #    a1 in val_authors or a2 in val_authors)
        seen_fandoms = (f1 in test_fandoms or f2 in test_fandoms)
        #f1 in val_fandoms or f2 in val_fandoms)
        if seen_fandoms:
            continue
        train_set_pairs.append(create_example(
            False, a1, a2, f1, f2, d1, d2
        ))
        train_fandoms|=set([f1, f2])
        train_authors|=set([a1, a2])
        num_da_train_pairs += 1

    print("  train fandoms", len(train_fandoms))
    print("  val fandoms", len(val_fandoms))
    print("  train_fandoms ^ test_fandoms = ", len(train_fandoms&test_fandoms))
    print("  train_fandoms ^ val_fandoms = ", len(train_fandoms&val_fandoms))
    print("  #authors in test set: ", len(train_authors))
    print("  #authors in val set: ", len(val_authors))
    print("  #authors train^val set: ", len(train_authors&val_authors))

    # saving examples to train, val and test .jsonl files
    save_examples_to_jsonl(path_to_train_jsonl, train_set_pairs)
    save_examples_to_jsonl(path_to_val_jsonl, val_set_pairs)
    save_examples_to_jsonl(path_to_test_jsonl, test_set_pairs)

    return train_set_pairs, val_set_pairs, test_set_pairs

def split_jsonl_dataset_into_train_val_test(path_to_original_jsonl: str,
                                            path_to_train_jsonl: str, 
                                            path_to_val_jsonl: str, 
                                            path_to_test_jsonl: str,
                                            split_function: Callable[[List[Dict], float], Tuple[List]],
                                            test_split_percentage: float):
    """
    Split the PAN dataset into train/val/test splits. This function is used with the
    *open_set_unseen_fandoms split functions.
    Args:
        path_to_original_jsonl (str): path to existing .jsonl file, such as ```pan20-av-large-no-text.jsonl```
        path_to_train_jsonl (str): path to .jsonl file where the training examples will be saved
        path_to_test_jsonl (str): path to .jsonl file where the test examples will be saved
        split_function (Callable): split function to be used:
            - split_pan_dataset_open_set_unseen_fandoms
            - split_pan_small_dataset_open_set_unseen_fandoms
        test_split_percentage: size of the Val+Test split as a percentage of the whole dataset
    """
    if os.path.exists(path_to_original_jsonl):
        examples = read_jsonl_examples(path_to_original_jsonl)
    else:
        print("File %s doesn't exist" % (path_to_original_jsonl))

    # split into train and test
    train_ids, val_ids, test_ids = split_function(
        examples=examples, 
        test_split_percentage=test_split_percentage
    )
    train_ids_map = {train_id:1 for train_id in train_ids}
    val_ids_map = {val_id:1 for val_id in val_ids}
    test_ids_map = {test_id:1 for test_id in test_ids}

    # saving examples to train, val and test .jsonl files
    print("Writing examples to %s, %s and %s" % (path_to_train_jsonl, path_to_val_jsonl, path_to_test_jsonl))
    with open(path_to_train_jsonl, "w") as f, open(path_to_val_jsonl, "w") as g, open(path_to_test_jsonl, "w") as h:
        for idx, example in enumerate(examples):
            if idx % 10000 == 0:
                print("[split_jsonl_dataset] Wrote %d examples" % (idx))
            if example['id'] in test_ids_map:
                json.dump(example, h)
                h.write('\n')
            elif example['id'] in val_ids_map:
                json.dump(example, g)
                g.write('\n')
            elif example['id'] in train_ids_map:
                json.dump(example, f)
                f.write('\n')


def split_jsonl_dataset(path_to_original_jsonl: str,
                        path_to_train_jsonl: str, 
                        path_to_test_jsonl: str,
                        split_function: Callable[[List[Dict], float], Tuple[List]],
                        test_split_percentage: float):
    """
    Split the PAN dataset into 2 splits. This wrapper function can be used to split the original dataset
    into train and test, as well as further split the train set into train and val.
    Args:
        path_to_original_jsonl (str): path to existing .jsonl file, such as ```pan20-av-large-no-text.jsonl```
        path_to_train_jsonl (str): path to .jsonl file where the training examples will be saved
        path_to_test_jsonl (str): path to .jsonl file where the test examples will be saved
        split_function (Callable): split function to be used:
            - split_pan_dataset_closed_set_v1
            - split_pan_dataset_closed_set_v2
            - split_pan_dataset_open_set_unseen_authors
        test_split_percentage: size of the Test split as a percentage of the whole dataset
    """
    if os.path.exists(path_to_original_jsonl):
        examples = read_jsonl_examples(path_to_original_jsonl)
    else:
        print("File %s doesn't exist" % (path_to_original_jsonl))

    # split into train and test
    train_ids, test_ids = split_function(
        examples=examples, 
        test_split_percentage=test_split_percentage
    )

    # saving examples to train and test .jsonl files
    print("Writing examples to %s and %s" % (path_to_train_jsonl, path_to_test_jsonl))
    with open(path_to_train_jsonl, "w") as f, open(path_to_test_jsonl, "w") as g:
        for idx, example in enumerate(examples):
            if idx % 10000 == 0:
                print("[split_jsonl_dataset] Wrote %d examples" % (idx))
            if example['id'] in test_ids:
                json.dump(example, g)
                g.write('\n')
            else:
                json.dump(example, f)
                f.write('\n')


def print_dataset_statistics(examples: List[Dict]):
    stats_dict = defaultdict(int)
    authors_dict = {}
    for example in examples:
        authors_dict[example['authors'][0]] = 1
        authors_dict[example['authors'][1]] = 1
        same_author = example['same']
        same_fandom = example['fandoms'][0] == example['fandoms'][1]
        if same_author:
            if same_fandom:
                stats_dict['sa_sf'] += 1
            else:
                stats_dict['sa_df'] += 1
        else:
            if same_fandom:
                stats_dict['da_sf'] += 1
            else:
                stats_dict['da_df'] += 1
        
    print("Dataset size: ", len(examples))
    print("    Same author pairs: ", stats_dict['sa_sf'] + stats_dict['sa_df'])
    print("        Same fandom pairs: ", stats_dict['sa_sf'])
    print("        Different fandom pairs: ", stats_dict['sa_df'])
    print("    Different author pairs: ", stats_dict['da_sf'] + stats_dict['da_df'])
    print("        Same fandom pairs: ", stats_dict['da_sf'])
    print("        Different fandom pairs: ", stats_dict['da_df'])
    print("Number of unique authors: ", len(authors_dict))


if __name__ == '__main__':
    # open split - unseen authors
    open_split_unseen_authors_paths = {
        "remote_xl_paths": {
            "data": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large.jsonl",
            "gt": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large-truth.jsonl",
            "original": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large.jsonl",
            "original_no_text": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-no-text.jsonl",
            "no_test": "/pan2020/open_splits/unseen_authors/xl/pan20-av-large-notest.jsonl",
            "train": "/pan2020/open_splits/unseen_authors/xl/pan20-av-large-train.jsonl",
            "val": "/pan2020/open_splits/unseen_authors/xl/pan20-av-large-val.jsonl",
            "test": "/pan2020/open_splits/unseen_authors/xl/pan20-av-large-test.jsonl"
        },
        "remote_xs_paths": {
            "data": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl",
            "gt": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl",
            "original": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small.jsonl",
            "original_no_text": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-no-text.jsonl",
            "no_test": "/pan2020/open_splits/unseen_authors/xs/pan20-av-small-notest.jsonl",
            "train": "/pan2020/open_splits/unseen_authors/xs/pan20-av-small-train.jsonl",
            "val": "/pan2020/open_splits/unseen_authors/xs/pan20-av-small-val.jsonl",
            "test": "/pan2020/open_splits/unseen_authors/xs/pan20-av-small-test.jsonl"
        }
    }
    
    # open split - unseen fandoms
    open_split_unseen_fandoms_paths = {
        "remote_xl_paths": {
            "data": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large.jsonl",
            "gt": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large-truth.jsonl",
            "original": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large.jsonl",
            "original_no_text": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-no-text.jsonl",
            "no_test": "/pan2020/open_splits/unseen_fandoms/xl/pan20-av-large-notest.jsonl",
            "train": "/pan2020/open_splits/unseen_fandoms/xl/pan20-av-large-train.jsonl",
            "val": "/pan2020/open_splits/unseen_fandoms/xl/pan20-av-large-val.jsonl",
            "test": "/pan2020/open_splits/unseen_fandoms/xl/pan20-av-large-test.jsonl"
        },
        "remote_xs_paths": {
            "data": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl",
            "gt": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl",
            "original": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small.jsonl",
            "original_no_text": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-no-text.jsonl",
            "no_test": "/pan2020/open_splits/unseen_fandoms/xs/pan20-av-small-notest.jsonl",
            "train": "/pan2020/open_splits/unseen_fandoms/xs/pan20-av-small-train.jsonl",
            "val": "/pan2020/open_splits/unseen_fandoms/xs/pan20-av-small-val.jsonl",
            "test": "/pan2020/open_splits/unseen_fandoms/xs/pan20-av-small-test.jsonl"
        }
    }

    # open split - unseen author and fandoms (test), unseen authors (val)
    open_split_unseen_all_paths = {
        "remote_xl_paths": {
            "data": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large.jsonl",
            "gt": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large-truth.jsonl",
            "original": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large.jsonl",
            "original_no_text": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-no-text.jsonl",
            "authors_dict": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-authors.json",
            "no_test": "/pan2020/open_splits/unseen_all/xl/pan20-av-large-notest.jsonl",
            "train": "/pan2020/open_splits/unseen_all/xl/pan20-av-large-train.jsonl",
            "val": "/pan2020/open_splits/unseen_all/xl/pan20-av-large-val.jsonl",
            "test": "/pan2020/open_splits/unseen_all/xl/pan20-av-large-test.jsonl"
        },
        "remote_xs_paths": {
            "data": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl",
            "gt": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl",
            "original": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small.jsonl",
            "original_no_text": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-no-text.jsonl",
            "authors_dict": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-authors.json",
            "no_test": "/pan2020/open_splits/unseen_all/xs/pan20-av-small-notest.jsonl",
            "train": "/pan2020/open_splits/unseen_all/xs/pan20-av-small-train.jsonl",
            "val": "/pan2020/open_splits/unseen_all/xs/pan20-av-small-val.jsonl",
            "test": "/pan2020/open_splits/unseen_all/xs/pan20-av-small-test.jsonl"
        },
        "remote_ner_xs_paths": {
            "train": "/pan2020/open_splits/unseen_all_ner/xs/pan20-av-small-noauthors-train.jsonl",
            "val": "/pan2020/open_splits/unseen_all_ner/xs/pan20-av-small-noauthors-val.jsonl",
            "test": "/pan2020/open_splits/unseen_all_ner/xs/pan20-av-small-noauthors-test.jsonl"
        }
    }

    # closed split v1
    closed_split_v1_paths = {  
        "remote_xl_paths": {
            "data": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large.jsonl",
            "gt": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large-truth.jsonl",
            "original": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large.jsonl",
            "original_no_text": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-no-text.jsonl",
            "no_test": "/pan2020/closed_splits/closed_split_v1/xl/pan20-av-large-notest.jsonl",
            "train": "/pan2020/closed_splits/closed_split_v1/xl/pan20-av-large-train.jsonl",
            "val": "/pan2020/closed_splits/closed_split_v1/xl/pan20-av-large-val.jsonl",
            "test": "/pan2020/closed_splits/closed_split_v1/xl/pan20-av-large-test.jsonl"
        },
        "remote_xs_paths": {
            "data": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl",
            "gt": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl",
            "original": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small.jsonl",
            "original_no_text": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-no-text.jsonl",
            "no_test": "/pan2020/closed_splits/closed_split_v1/xs/pan20-av-small-notest.jsonl",
            "train": "/pan2020/closed_splits/closed_split_v1/xs/pan20-av-small-train.jsonl",
            "val": "/pan2020/closed_splits/closed_split_v1/xs/pan20-av-small-val.jsonl",
            "test": "/pan2020/closed_splits/closed_split_v1/xs/pan20-av-small-test.jsonl"
        },
        "remote_ner_xs_paths": {
            "train": "/pan2020/closed_splits/closed_split_v1_ner/xs/pan20-av-small-noauthors-train.jsonl",
            "val": "/pan2020/closed_splits/closed_split_v1_ner/xs/pan20-av-small-noauthors-val.jsonl",
            "test": "/pan2020/closed_splits/closed_split_v1_ner/xs/pan20-av-small-noauthors-test.jsonl"
        },
        "local_xl_paths": {
            "original": "../data/pan2020_xl/pan20-av-large.jsonl",
            "no_test": "../data/pan2020_xl/pan20-av-large-notest.jsonl",
            "train": "../data/pan2020_xl/pan20-av-large-train.jsonl",
            "val": "../data/pan2020_xl/pan20-av-large-val.jsonl",
            "test": "../data/pan2020_xl/pan20-av-large-test.jsonl"
        },
        "local_xs_paths": {
            "original": "../data/pan2020_xs/pan20-av-small.jsonl",
            "no_test": "../data/pan2020_xs/pan20-av-small-notest.jsonl",
            "train": "../data/pan2020_xs/pan20-av-small-train.jsonl",
            "val": "../data/pan2020_xs/pan20-av-small-val.jsonl",
            "test": "../data/pan2020_xs/pan20-av-small-test.jsonl"
        }
    }

    # closed split v2
    closed_split_v2_paths = {  
        "remote_xl_paths": {
            "data": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large.jsonl",
            "gt": "/pan2020/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large-truth.jsonl",
            "original": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large.jsonl",
            "original_no_text": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-no-text.jsonl",
            "authors_dict": "/pan2020/pan20-authorship-verification-training-large/pan20-av-large-authors.json",
            "no_test": "/pan2020/closed_splits/closed_split_v2/xl/pan20-av-large-notest.jsonl",
            "train": "/pan2020/closed_splits/closed_split_v2/xl/pan20-av-large-train.jsonl",
            "val": "/pan2020/closed_splits/closed_split_v2/xl/pan20-av-large-val.jsonl",
            "test": "/pan2020/closed_splits/closed_split_v2/xl/pan20-av-large-test.jsonl"
        },
        "remote_xs_paths": {
            "data": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl",
            "gt": "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl",
            "original": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small.jsonl",
            "original_no_text": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-no-text.jsonl",
            "authors_dict": "/pan2020/pan20-authorship-verification-training-small/pan20-av-small-authors.json",
            "no_test": "/pan2020/closed_splits/closed_split_v2/xs/pan20-av-small-notest.jsonl",
            "train": "/pan2020/closed_splits/closed_split_v2/xs/pan20-av-small-train.jsonl",
            "val": "/pan2020/closed_splits/closed_split_v2/xs/pan20-av-small-val.jsonl",
            "test": "/pan2020/closed_splits/closed_split_v2/xs/pan20-av-small-test.jsonl"
        },
        "local_xl_paths": {
            "original": "../data/pan2020_xl/pan20-av-large.jsonl",
            "no_test": "../data/pan2020_xl/pan20-av-large-notest.jsonl",
            "train": "../data/pan2020_xl/pan20-av-large-train.jsonl",
            "val": "../data/pan2020_xl/pan20-av-large-val.jsonl",
            "test": "../data/pan2020_xl/pan20-av-large-test.jsonl"
        },
        "local_xs_paths": {
            "original": "../data/pan2020_xs/pan20-av-small.jsonl",
            "no_test": "../data/pan2020_xs/pan20-av-small-notest.jsonl",
            "train": "../data/pan2020_xs/pan20-av-small-train.jsonl",
            "val": "../data/pan2020_xs/pan20-av-small-val.jsonl",
            "test": "../data/pan2020_xs/pan20-av-small-test.jsonl"
        }
    }
    
    # TODO: change with the appropriate paths
    paths_dict = closed_split_v1_paths['remote_ner_xs_paths']

    # Step 1: split original dataset into:
    #   - Train (pan20-av-*-notest.jsonl)
    #   - Test (pan20-av-*-test.jsonl)
    # We already have the train/test splits, skip this
    # split_jsonl_dataset(
    #     path_to_original_jsonl=paths_dict['original'],
    #     path_to_train_jsonl=paths_dict['no_test'],
    #     path_to_test_jsonl=paths_dict['test'],
    #     split_function=split_pan_dataset_open_set_unseen_fandoms,
    #     test_split_percentage=0.05
    # )

    # TODO Step 2: split Train dataset into Train and Val
    # split_jsonl_dataset(
    #     path_to_original_jsonl=paths_dict['no_test'],
    #     path_to_train_jsonl=paths_dict['train'],
    #     path_to_test_jsonl=paths_dict['val'],
    #     split_function=split_pan_dataset_closed_set_v2,
    #     test_split_percentage=0.1
    # )

    # split original dataset into Train/Val/Test
    # this function is used to create all 3 splits simultaneously for the open splits
    # of unseen fandoms. This is due to the test fandoms needing to be the same as 
    # the val fandoms.
    # split_jsonl_dataset_into_train_val_test(
    #     path_to_original_jsonl=paths_dict['original'],
    #     path_to_train_jsonl=paths_dict['train'],
    #     path_to_val_jsonl=paths_dict['val'],
    #     path_to_test_jsonl=paths_dict['test'],
    #     split_function=split_pan_dataset_open_set_unseen_fandoms,
    #     test_split_percentage=0.1
    # )

    # Step 3: write .jsonl files to folders 
    write_jsonl_to_folder(paths_dict['train'], paths_dict['train'].replace(".jsonl", ""))
    write_jsonl_to_folder(paths_dict['val'], paths_dict['val'].replace(".jsonl", ""))
    write_jsonl_to_folder(paths_dict['test'], paths_dict['test'].replace(".jsonl", ""))
    
    # tokenizer = BertTokenizer.from_pretrained(
    #     os.path.join('..', 'pretrained_models', 'bert-base-uncased'),
    #     do_lower_case=True
    # )
    # print("Tokenizer = ", tokenizer)
    # tokenizer = tokenizer.basic_tokenizer
    # print("Tokenizer = ", tokenizer)

    # # Ignore lines below
    # examples = read_jsonl_examples(remote_xl_paths['original'], 1000)
    # examples_length = []
    # for idx, example in enumerate(examples):
    #     print("idx = ", idx)
    #     tokens_a = tokenizer.tokenize(example["pair"][0])
    #     tokens_b = tokenizer.tokenize(example["pair"][1])
    #     examples_length.append(len(tokens_a)+len(tokens_b))
    # lens = np.array(examples_length)
    # print("Avg length = ", np.mean(lens))
    # print("Max length = ", np.max(lens))
    # print("Min length = ", np.min(lens))
    # print("Std length = ", np.std(lens))

    #print_dataset_statistics(examples)
    #(train_ids, test_ids) = split_pan_dataset_open_set_unseen_fandoms(examples, 0.1)
    #(train_ids, test_ids) = split_pan_small_dataset_open_set_unseen_fandoms(examples, 0.1)

    #train_examples = read_jsonl_examples('../data/pan2020_xl/pan20-av-large.jsonl')
    #train_examples = read_jsonl_examples('../data/pan2020_xs/pan20-av-small.jsonl')
    #split_pan_dataset_open_set_unseen_authors(train_examples, 0.05)

    # root_dir = paths_dict['test'].replace(".jsonl", "")
    # for idx, fname in enumerate(os.listdir(root_dir)):
    #     if idx % 1000 == 0:
    #         print(idx)
    #     with open(os.path.join(root_dir, fname)) as fp:
    #         try:
    #             entry = json.load(fp)
    #         except:
    #             print("error at ", fname)

    #         assert 'same' in entry, " missing key in %s" % (fname)
    #         assert 'authors' in entry, " missing key in %s" % (fname)
    #         assert 'id' in entry, " missing key in %s" % (fname)
    #         assert 'fandoms' in entry, " missing key in %s" % (fname)
    #         assert 'pair' in entry, " missing key in %s" % (fname)
 
    # train_examples = read_jsonl_examples(paths_dict['train'])
    # val_examples = read_jsonl_examples(paths_dict['val'])
    # test_examples = read_jsonl_examples(paths_dict['test'])

    # train_examples, val_examples, test_examples = split_jsonl_dataset_resampling(
    #     path_to_original_jsonl=None,
    #     path_to_authors_json=paths_dict['authors_dict'],
    #     path_to_train_jsonl=paths_dict['train'],
    #     path_to_val_jsonl=paths_dict['val'],
    #     path_to_test_jsonl=paths_dict['test'],
    #     train_size=248000,
    #     test_size=13700
    # )

    # statistics
    # def get_stats(examples):
    #     stats_dict = defaultdict(int)
    #     fandoms_set = set()
    #     authors_set = set()
    #     stats_dict['size'] = len(examples)
    #     for example in examples:
    #         f1, f2 = example['fandoms'][0], example['fandoms'][1]
    #         a1, a2 = example['authors'][0], example['authors'][1]
    #         same_author = example['same']
    #         same_fandom = f1 == f2
    #         fandoms_set|=set([f1, f2])
    #         authors_set|=set([a1, a2])

    #         if same_author:
    #             if same_fandom:
    #                 stats_dict['sa_sf'] += 1
    #             else:
    #                 stats_dict['sa_df'] += 1
    #         else:
    #             if same_fandom:
    #                 stats_dict['da_sf'] += 1
    #             else:
    #                 stats_dict['da_df'] += 1

    #     return stats_dict, authors_set, fandoms_set

    # train_stats, train_authors, train_fandoms = get_stats(train_examples)
    # val_stats, val_authors, val_fandoms = get_stats(val_examples)
    # test_stats, test_authors, test_fandoms = get_stats(test_examples)
    
    # for split_name, stats_dict in zip(["TRAIN", "VAL", "TEST"], [train_stats, val_stats, test_stats]):
    #     print("%s size: %d" % (split_name, stats_dict['size']))
    #     print("    Same author pairs: ", stats_dict['sa_sf'] + stats_dict['sa_df'])
    #     print("        Same fandom pairs: ", stats_dict['sa_sf'])
    #     print("        Different fandom pairs: ", stats_dict['sa_df'])
    #     print("    Different author pairs: ", stats_dict['da_sf'] + stats_dict['da_df'])
    #     print("        Same fandom pairs: ", stats_dict['da_sf'])
    #     print("        Different fandom pairs: ", stats_dict['da_df'])

    # print("train_authors ^ test_authors:", len(train_authors&test_authors))
    # print("val_authors ^ test_authors:", len(val_authors&test_authors))
    # print("train_authors ^ val_authors:", len(train_authors&val_authors))

    # print("train_fandoms ^ test_fandoms:", len(train_fandoms&test_fandoms))
    # print("val_fandoms ^ test_fandoms:", len(val_fandoms&test_fandoms))
    # print("train_fandoms ^ val_fandoms:", len(train_fandoms&val_fandoms))