import os
import json
import pairing.data
import collections
import tqdm


FOLD_DIR = "dataset/folds"

def make_pairdata(dataset,covered_notes):
    all_notes_set = set(covered_notes)
    for d in dataset:
        if not d["blend_notes"]:
            # This datapoint should have been cleaned, sorry.
            continue
        if not set(d["blend_notes"]).intersection(all_notes_set):
            # This datapoint contains a note that appears in only one of {train,test}
            continue

        pair_input = (d["edge"],collections.Counter(d["blend_notes"]))
        try:
            yield pairing.data.to_pairdata(pair_input,covered_notes)
        except AttributeError:
            continue

def make_fold(fname):
    with open(os.path.join(FOLD_DIR,fname)) as f:
        dataset = json.load(f)

    covered_notes = dataset["covered_notes"]
    train = list(make_pairdata(dataset["train"],covered_notes))
    test = list(make_pairdata(dataset["test"],covered_notes))
    return {"train":train, "test":test, "covered_notes": covered_notes} 

def load_fold_datasets():
    all_data = []
    for fold_fname in os.listdir(FOLD_DIR):
        print(f"Reading from {fold_fname}")
        all_data.append(make_fold(fold_fname))
    return all_data