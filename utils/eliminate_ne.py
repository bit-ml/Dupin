#%%
from spacy import displacy
import en_core_web_trf
import json

example_path = "/darkweb_ds/open_splits/unseen_authors/xs/pan20-av-small-test/aa69227b-f768-586c-9bff-9ae5105e6873.json"
dataset_path = "/darkweb_ds/reddit_darknet/train"

nlp = en_core_web_trf.load()
ex = json.load(open(example_path))
doc1 = ex["pair"][0]

def get_named_entities(text: str) -> list:
    processed = nlp(text)
    print(type(processed))
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


named_entities_list = get_named_entities(doc1)
entities_to_be_removed = get_entity_subset(named_entities_list, ["PERSON"])
new_text = remove_named_entities(doc1, entities_to_be_removed)
print(new_text)

# %%
