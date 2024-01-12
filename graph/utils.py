import torch
import json


full_canonize = dict()

CANONIZE_DICTIONARY = {
    "":None,
    "No odor group found for these":None,
    "anisic":"anise",
    "corn chip":"corn",
    "medicinal,":"medicinal",
}
full_canonize.update(CANONIZE_DICTIONARY)

# # If you want to try carving all notes, you can comment this section.
# FAILED_TO_CARVE = {n:None for n in ['acetic', 'ammoniacal', 'cabbage', 'celery', 'cherry', 'corn', 'dusty', 'eggy', 'fishy', 'garlic', 'hay', 'juicy', 'lactonic', 'leathery', 'malty', 'moldy', 'mushroom', 'peach', 'potato', 'pungent', 'rooty', 'salty', 'smoky', 'tarragon', 'toasted', 'tobacco', 'tomato']}
# full_canonize.update(FAILED_TO_CARVE)

def canonize_note(note):
    while note in full_canonize:
        note = full_canonize[note]
    return note

def canonize(notes):
    cleaned = set()
    for n in notes:
        canonized = canonize_note(n)
        if canonized:
            cleaned.add(canonized)
    return sorted(list(cleaned))

ALL_NOTES_LIST = ['', 'No odor group found for these', 'acetic', 'acidic', 'alcoholic', 'aldehydic', 'alliaceous', 'amber', 'ammoniacal', 'animal', 'anise', 'anisic', 'aromatic', 'balsamic', 'berry', 'bitter', 'bready', 'brown', 'burnt', 'buttery', 'cabbage', 'camphoreous', 'caramellic', 'celery', 'cheesy', 'chemical', 'cherry', 'chocolate', 'citrus', 'clean', 'cocoa', 'coconut', 'coffee', 'cooling', 'corn', 'corn chip', 'coumarinic', 'creamy', 'dairy', 'dusty', 'earthy', 'eggy', 'estery', 'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruity', 'fungal', 'fusel', 'garlic', 'green', 'hay', 'herbal', 'honey', 'jammy', 'juicy', 'lactonic', 'leathery', 'licorice', 'malty', 'marine', 'meaty', 'medicinal', 'medicinal,', 'melon', 'mentholic', 'minty', 'moldy', 'mossy', 'mushroom', 'musk', 'mustard', 'musty', 'nutty', 'oily', 'onion', 'orris', 'peach', 'phenolic', 'pine', 'potato', 'powdery', 'pungent', 'roasted', 'rooty', 'rummy', 'salty', 'smoky', 'soapy', 'solvent', 'sour', 'spicy', 'sulfurous', 'sweet', 'tarragon', 'thujonic', 'toasted', 'tobacco', 'tomato', 'tonka', 'tropical', 'vanilla', 'vegetable', 'waxy', 'winey', 'woody']
CANON_NOTES_LIST = canonize(ALL_NOTES_LIST)
canon_notes_set = set(CANON_NOTES_LIST)

for n in CANON_NOTES_LIST:
    assert not n in full_canonize

def multi_hot(notes,underyling_list=None,should_canonize=False):
    if should_canonize:
        notes = canonize(notes)
    if not underyling_list:
        underyling_list = CANON_NOTES_LIST
    notes = [n for n in notes if n in underyling_list]
    indices = torch.tensor([underyling_list.index(n) for n in notes])
    if len(indices) == 0:
        return torch.zeros(len(underyling_list))
        # Occurs when the notes in the pair were removed due to infrequency.
        # raise AttributeError("Found no valid notes.")
    one_hots = torch.nn.functional.one_hot(indices, len(underyling_list))
    return one_hots.sum(dim=0).float()

def sort(m1,m2):
    if m1 > m2:
        return (m1,m2)
    else:
        return (m2,m1)

def missing_notes(notes):
    return canon_notes_set.difference(notes)

if __name__ == "__main__":
    with open("dataset/full.json") as f:
        full_data = json.load(f)

    all_notes = set()
    for d in full_data:
        all_notes.update(d["blend_notes"])
    print(sorted(all_notes))
    print(CANON_NOTES_LIST)