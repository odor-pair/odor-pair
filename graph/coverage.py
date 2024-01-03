import json
import random
import collections
import tqdm
import json
import graph.utils

with open("dataset/full.json") as f:
    full_data = json.load(f)

all_edges = set()
all_nodes = set()
edge_to_notes = dict()

for d in full_data:
    edge = graph.utils.sort(d["mol1"],d["mol2"])
    all_edges.add(edge)
    edge_to_notes[edge] = graph.utils.canonize(d["blend_notes"])

    all_nodes.add(d["mol1"])
    all_nodes.add(d["mol2"])


for n1, n2 in all_edges:
    assert not (n2,n1) in all_edges
    assert not n1 == n2

full_graph = collections.defaultdict(set)
for n1, n2 in all_edges:
    full_graph[n1].add(n2)
    full_graph[n2].add(n1)

train_fraction = .6
test_fraction = .4

assert train_fraction + test_fraction == 1

def build_edges(all_nodes):
    all_edges = set()
    for node in all_nodes:
        for other in full_graph[node]:
            if not other in all_nodes:
                continue
            all_edges.add(graph.utils.sort(node,other))
    return all_edges

def random_split_carving():
    train_nodes = set(random.sample(sorted(all_nodes),int(len(all_nodes)*train_fraction)))
    test_nodes = all_nodes.difference(train_nodes)
    
    train_edges = build_edges(train_nodes)
    test_edges = build_edges(test_nodes)

    assert not train_nodes.intersection(test_nodes)
    assert not train_edges.intersection(test_edges)
    
    return train_edges, test_edges

def get_covered_notes(edges):
    covered = set()
    for edge in edges:
        covered.update(edge_to_notes[edge])
    return covered



all_covered = set()
missing = graph.utils.missing_notes(all_covered)
cross_validation_folds = []
i = 0
while missing:
    i += 1
    print(i)
    trn_edges, tst_edges = random_split_carving()
    # print(i,len(trn_edges),len(tst_edges))
    valid_covered_notes = get_covered_notes(trn_edges).intersection(get_covered_notes(tst_edges))
    new_covered = valid_covered_notes.difference(all_covered)
    if new_covered:
        all_covered.update(valid_covered_notes)
        print(f"We are covering {len(all_covered)} but still missing {len(graph.utils.missing_notes(all_covered))}")
        missing = graph.utils.missing_notes(all_covered)
        cross_validation_folds.append((trn_edges,tst_edges))


print(f"Covered all notes in {len(cross_validation_folds)} folds")

def make_dataset(edges):
    return [{"edge":edge,"blend_notes":edge_to_notes[edge]} for edge in edges]

for i, (fold_train, fold_test) in enumerate(cross_validation_folds):
    covered = get_covered_notes(fold_train).intersection(get_covered_notes(fold_test))
    result = {"train":make_dataset(fold_train),"test":make_dataset(fold_test),"covered_notes":sorted(list(covered))}
    with open(f"dataset/folds/fold{i}.json","w") as f:
        json.dump(result,f)
