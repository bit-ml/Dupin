#%%
from spacy import displacy
import en_core_web_trf
import json


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


def remove_entities_from_json(path: str, spacy_ner_f=en_core_web_trf) -> dict:
    spacy_ner_f = spacy_ner_f.load()
    dataset = json.load(open(path))

    text_a1 = dataset["pair"][0]
    text_a2 = dataset["pair"][1]

    ne_t1 = get_named_entities(text_a1, spacy_ner_f)
    ne_t2 = get_named_entities(text_a2, spacy_ner_f)

    entities_to_be_removed_t1 = get_entity_subset(ne_t1, ["PERSON"])
    entities_to_be_removed_t2 = get_entity_subset(ne_t2, ["PERSON"])

    dataset["pair"][0] = remove_named_entities(text_a1, entities_to_be_removed_t1)
    dataset["pair"][1] = remove_named_entities(text_a2, entities_to_be_removed_t2)

    return dataset


example_path = "/darkweb_ds/open_splits/unseen_authors/xs/pan20-av-small-test/aa69227b-f768-586c-9bff-9ae5105e6873.json"
ds = remove_entities_from_json(example_path)
print(ds)
