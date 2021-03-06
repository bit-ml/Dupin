#%%
import en_core_web_trf
import json
import os
import time
from pathlib import Path
from spacy import displacy
from typing import Dict
import spacy
spacy.require_gpu()
#nlp = spacy.load('en_core_web_md')


def get_named_entities(text: str, spacy_ner_f) -> list:
    processed = spacy_ner_f(text)
    named_entities = []

    for ent in processed.ents:
        named_entities.append((ent.text, ent.label_))

    named_entities = set(named_entities)

    return named_entities


def get_entity_subset(entity_list: list, entity_names: list) -> list:
    entity_subset = []
    for (str_, entity) in entity_list:
        if entity in entity_names:
            entity_subset.append((str_, entity))

    return entity_subset


def remove_named_entities(text: str, entity_subset: list) -> str:
    for (str_, entity) in entity_subset:
        text = text.replace(str_, entity)
    return text


def remove_entities_from_json(
    sample: Dict, spacy_ner_f, removed_entities=["PERSON"], 
) -> dict:
    #spacy_ner_f = spacy_ner_f.load()

    text_a1 = sample["pair"][0]
    text_a2 = sample["pair"][1]

    tic1 = time.perf_counter()
    ne_t1 = get_named_entities(text_a1, spacy_ner_f)
    ne_t2 = get_named_entities(text_a2, spacy_ner_f)

    tic2 = time.perf_counter()
    entities_to_be_removed_t1 = get_entity_subset(ne_t1, removed_entities)
    entities_to_be_removed_t2 = get_entity_subset(ne_t2, removed_entities)

    tic3 = time.perf_counter()
    sample["pair"][0] = remove_named_entities(text_a1, entities_to_be_removed_t1)
    sample["pair"][1] = remove_named_entities(text_a2, entities_to_be_removed_t2)
    tic4 = time.perf_counter()

    # print("time of get_named_entities(): ", (tic2-tic1))
    # print("time of get_entity_subset(): ", (tic3-tic2))
    # print("time of remove_named_entities(): ", (tic4-tic3))

    return sample


def main():
    dataset = 'reddit'
    if dataset == 'pan':
        modes = ["train", "val", "test"]
        # root = "/darkweb_ds/"
        # subset_type = "open_splits/unseen_authors/xs/"
        root = "/pan2020"
        #subset_type = "open_splits/unseen_all/xl"
        subset_type = "closed_splits/closed_split_v1/xs"
        spacy_ner_f = spacy.load('en_core_web_trf')

        for mode in modes:
            print(f"[processing] Mode: {mode}")
            #subset_mode = f"pan20-av-large-{mode}.jsonl"
            #removed_ne_path = f"pan20-av-large-noauthors-{mode}.jsonl"
            subset_mode = f"pan20-av-small-{mode}.jsonl"
            removed_ne_path = f"pan20-av-small-noauthors-{mode}.jsonl"

            ds_path_full = os.path.join(root, subset_type, subset_mode)
            ds_path_removed_ne = os.path.join(root, subset_type, removed_ne_path)
            print("ds path: ", ds_path_full)
            print("target path: ", ds_path_removed_ne)

            #Path(ds_path_removed_ne).mkdir(parents=True, exist_ok=True)
            new_samples = []
            if ds_path_full.endswith('.jsonl'):
                with open(ds_path_full) as f:
                    for idx, line in enumerate(f):
                        print("processed ", idx, " examples")
                        sample = json.loads(line)
                        new_samples.append(remove_entities_from_json(
                            sample=sample, 
                            spacy_ner_f=spacy_ner_f
                        ))
                        # if idx == 3:
                        #     break

            with open(ds_path_removed_ne, "w") as f:
                for sample in new_samples:
                    json.dump(sample, f)
                    f.write("\n")
    elif dataset == 'reddit':
        modes = ["train", "val", "test"]
        root = "/pan2020"
        subset_type = "reddit_darknet/reddit_open_split_50per"
        subset_type_ner = subset_type + "_ner"
        spacy_ner_f = spacy.load('en_core_web_trf')

        for mode in modes:
            print(f"[processing] Mode: {mode}")
            ds_path_full = os.path.join(root, subset_type, mode)
            ds_path_removed_ne = os.path.join(root, subset_type_ner, mode)
            print("ds path: ", ds_path_full)
            print("target path: ", ds_path_removed_ne)
            if not os.path.exists(ds_path_removed_ne):
                os.makedirs(ds_path_removed_ne)

            fnames = [fn for fn in os.listdir(ds_path_full) if fn.endswith('.json')]
            for idx, fn in enumerate(fnames):
                print("Processed %d files " % (idx))
                fn_path = os.path.join(ds_path_full, fn)
                with open(fn_path) as f:
                    sample = json.load(f)
                    new_sample = remove_entities_from_json(
                        sample=sample, 
                        spacy_ner_f=spacy_ner_f
                    )
                fn_path = os.path.join(ds_path_removed_ne, fn)
                with open(fn_path, "w") as f:
                    json.dump(new_sample, f, indent=2)
                    #f.write('\n')
                #break

main()
# %%
