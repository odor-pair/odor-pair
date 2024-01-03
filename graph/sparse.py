import json
import random
import collections
import tqdm
import json
import graph.stats
import graph.utils

with open("dataset/full.json") as f:
    full_data = json.load(f)

def sort(m1,m2):
    if m1 > m2:
        return (m1,m2)
    else:
        return (m2,m1)

edges = set()
nodes = set()
for d in full_data:
    edges.add(sort(d["mol1"],d["mol2"]))
    nodes.add(d["mol1"])
    nodes.add(d["mol2"])

notes_counter = collections.Counter()
notes_to_mols = collections.defaultdict(set)
for d in full_data:
    notes_counter.update(d["blend_notes"])
    for note in d["blend_notes"]:
        notes_to_mols[note].add(d["mol1"])
        notes_to_mols[note].add(d["mol2"])

print("The following notes do not have sufficient data:")
for note, mols in notes_to_mols.items():
    if len(mols) < 10 or notes_counter[note] < 10:
        print(note)