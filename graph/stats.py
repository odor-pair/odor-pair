import json
import random
import collections
import tqdm
import json
import torch
import graph.utils
import scipy
import numpy as np

with open("dataset/full.json") as f:
        full_data = json.load(f)

full_data = {graph.utils.sort(d["mol1"],d["mol2"]):d for d in full_data}

def get_note_frequencies(edges):
    edge_sum = []
    for edge in edges:
        if not tuple(edge) in full_data:
            continue
        data = full_data[tuple(edge)]
        edge_sum.append(graph.utils.multi_hot(data["blend_notes"],should_canonize=True))
    return torch.stack(edge_sum).sum(axis=0)

full_freq = get_note_frequencies(full_data.keys())
# train_freq = get_note_frequencies(train_edges)
# test_freq = get_note_frequencies(test_edges)

def print_freqs(freqs):
    for k, f in list(zip(graph.utils.CANON_NOTES_LIST,freqs)):
        print(k,f)

def kl_div(source,target,eps=1e-12):
    src_nrm = (source + eps) / (source.sum() + eps)
    trg_nrm = (target + eps) / (target.sum() + eps)
    
    assert np.isclose(src_nrm.sum(),1)
    assert np.isclose(trg_nrm.sum(),1)
    divergence = scipy.special.kl_div(src_nrm,trg_nrm)
    return divergence.sum()

k = 1

def kl_similarity(edges,target_edges=None):
    freqs = get_note_frequencies(edges)
    if target_edges is None:
        target_freq = full_freq
    else:
        target_freq = get_note_frequencies(target_edges)

    return np.exp(-k*kl_div(freqs,target_freq))

def count_nonzero(edges):
    freqs = get_note_frequencies(edges)
    return torch.count_nonzero(freqs)

def live_split():
    with open("dataset/old/by_stats.json") as f:
        carving = json.load(f)

    train_edges = carving["train"]
    print(len(train_edges))
    print(kl_similarity(train_edges))

    test_edges = carving["test"]
    print(len(test_edges))
    print(kl_similarity(test_edges))

    failed = 0
    for n, ff, trnf, tstf in list(zip(graph.utils.CANON_NOTES_LIST,full_freq.numpy(),get_note_frequencies(train_edges).numpy(),get_note_frequencies(test_edges).numpy())):
        assert ff
        print(n,ff)
        # if trnf < 1 or tstf < 1:
        #     print(f"{n} w/ {ff:.0f}. Appears in train {trnf:.0f} and test {tstf:.0f}")
        #     failed += 1

    print(f"Out of {len(graph.utils.CANON_NOTES_LIST)}, but found {failed} w/o enough data.")
    print(count_nonzero(train_edges))
    print(count_nonzero(test_edges))

if __name__ == "__main__":
    live_split()