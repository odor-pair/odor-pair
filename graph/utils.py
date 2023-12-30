import torch

CANONIZE_DICTIONARY = {"":None,"No odor group found for these":None,
    "anisic":"anise","corn chip":"corn",
    "medicinal,":"medicinal"}
CANON_NOTES_LIST = ['acetic', 'acidic', 'alcoholic', 'aldehydic', 'alliaceous', 'amber', 'ammoniacal', 'animal', 'anise', 'aromatic', 'balsamic', 'berry', 'bitter', 'bready', 'brown', 'burnt', 'buttery', 'cabbage', 'camphoreous', 'caramellic', 'celery', 'cheesy', 'chemical', 'cherry', 'chocolate', 'citrus', 'clean', 'cocoa', 'coconut', 'coffee', 'cooling', 'corn', 'coumarinic', 'creamy', 'dairy', 'dusty', 'earthy', 'eggy', 'estery', 'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruity', 'fungal', 'fusel', 'garlic', 'green', 'hay', 'herbal', 'honey', 'jammy', 'juicy', 'lactonic', 'leathery', 'licorice', 'malty', 'marine', 'meaty', 'medicinal', 'melon', 'mentholic', 'minty', 'moldy', 'mossy', 'mushroom', 'musk', 'mustard', 'musty', 'nutty', 'oily', 'onion', 'orris', 'peach', 'phenolic', 'pine', 'potato', 'powdery', 'pungent', 'roasted', 'rooty', 'rummy', 'salty', 'smoky', 'soapy', 'solvent', 'sour', 'spicy', 'sulfurous', 'sweet', 'tarragon', 'thujonic', 'toasted', 'tobacco', 'tomato', 'tonka', 'tropical', 'vanilla', 'vegetable', 'waxy', 'winey', 'woody']

def canonize(notes):
    cleaned = set()
    for n in notes:
        # Valid note
        if not n in CANONIZE_DICTIONARY:
            cleaned.add(n)
        # Should be removed
        elif not CANONIZE_DICTIONARY[n]:
            continue
        else:
            cleaned.add(CANONIZE_DICTIONARY[n])
    return cleaned

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