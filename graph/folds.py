import os
import json
import pairing.data
import collections
import tqdm


FOLD_DIR = "dataset/folds"

def make_pairdata(dataset,covered_notes):
    all_notes_set = set(covered_notes)
    frequencies = {n:0 for n in covered_notes}
    datapoints = []
    for d in dataset:
        if not d["blend_notes"]:
            # This datapoint should have been cleaned, sorry.
            continue
        if not set(d["blend_notes"]).intersection(all_notes_set):
            # This datapoint contains a note that appears in only one of {train,test}
            continue

        pair_input = (d["edge"],collections.Counter(d["blend_notes"]))
        try:
            datapoints.append(pairing.data.to_pairdata(pair_input,covered_notes))
            for note in d["blend_notes"]:
                if note in covered_notes:
                    frequencies[note] += 1
        except AttributeError:
            continue

    return datapoints, frequencies

def make_fold(fname):
    with open(fname) as f:
        dataset = json.load(f)

    covered_notes = dataset["covered_notes"]
    train, trnf = make_pairdata(dataset["train"],covered_notes)
    test, tstf = make_pairdata(dataset["test"],covered_notes)
    return {"train":train, "test":test, "covered_notes": covered_notes, "train_frequencies":trnf, "test_frequencies":tstf}

def load_fold_datasets():
    all_data = []
    for fold_fname in os.listdir(FOLD_DIR):
        print(f"Reading from {fold_fname}")
        all_data.append(make_fold(os.path.join(FOLD_DIR,fname)))
    return all_data

def load_single_fold(fname="dataset/single_fold.json"):
    return [make_fold(fname)]