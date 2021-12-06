#%%
import en_core_web_trf
import json
import os
from pathlib import Path
from spacy import displacy


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
    path: str, removed_entities=["PERSON"], spacy_ner_f=en_core_web_trf
) -> dict:
    spacy_ner_f = spacy_ner_f.load()
    sample = json.load(open(path))

    text_a1 = sample["pair"][0]
    text_a2 = sample["pair"][1]

    ne_t1 = get_named_entities(text_a1, spacy_ner_f)
    ne_t2 = get_named_entities(text_a2, spacy_ner_f)

    entities_to_be_removed_t1 = get_entity_subset(ne_t1, removed_entities)
    entities_to_be_removed_t2 = get_entity_subset(ne_t2, removed_entities)

    sample["pair"][0] = remove_named_entities(text_a1, entities_to_be_removed_t1)
    sample["pair"][1] = remove_named_entities(text_a2, entities_to_be_removed_t2)

    return sample


def main():
    modes = ["train", "test"]
    root = "/darkweb_ds/"
    subset_type = "open_splits/unseen_authors/xs/"

    for mode in modes:
        print(f"[processing] Mode: {mode}")
        subset_mode = f"/pan20-av-small-{mode}/"
        removed_ne_path = f"/pan20-av-small-noauthors-{mode}"

        ds_path_full = os.path.join(root, subset_type, subset_mode)
        ds_path_removed_ne = os.path.join(root, subset_type, removed_ne_path)

        Path(ds_path_removed_ne).mkdir(parents=True, exist_ok=True)

        for fname in os.listdir(ds_path_full):
            if fname.endswith(".json"):
                sample = remove_entities_from_json(fname)
                with open(os.path.join(ds_path_removed_ne, fname)) as json_fp:
                    json.dump(sample, json_fp, indent=2)

main()
# %%
