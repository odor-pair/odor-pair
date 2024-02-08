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

train_fraction = .5
test_fraction = .5

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

def random_split_triple_carving(trnf,tstf,vldf):
    assert trnf + tstf + vldf == 1
    shuffled_nodes = list(random.sample(sorted(all_nodes), len(all_nodes)))

    trns, tsts, vlds = int(trnf * len(all_nodes)), int(tstf * len(all_nodes)), int(vldf * len(all_nodes))

    train_nodes = set(shuffled_nodes[:trns])
    test_nodes = set(shuffled_nodes[trns:trns+tsts])
    validate_nodes = set(shuffled_nodes[trns+tsts:])

    train_edges = build_edges(train_nodes)
    test_edges = build_edges(test_nodes)
    validate_edges = build_edges(validate_nodes)

    assert not train_nodes.intersection(test_nodes)
    assert not train_nodes.intersection(validate_nodes)
    assert not test_nodes.intersection(validate_nodes)

    assert not train_edges.intersection(test_edges)
    assert not train_edges.intersection(validate_edges)
    assert not test_edges.intersection(validate_edges)
    
    return train_edges, test_edges, validate_edges

def get_covered_notes(edges):
    covered = set()
    for edge in edges:
        covered.update(edge_to_notes[edge])
    return covered

def make_dataset(edges):
    return [{"edge":edge,"blend_notes":edge_to_notes[edge]} for edge in edges]

def make_full_coverage_folds():
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

    for i, (fold_train, fold_test) in enumerate(cross_validation_folds):
        covered = get_covered_notes(fold_train).intersection(get_covered_notes(fold_test))
        # "covered_notes" only has notes that appear in both the train and test dataset
        # There may be notes that appear only in the train or only in the test dataset
        # so they are not in covered notes. Also, it may not be the case that a datapoint has any notes in covered notes.
        result = {"train":make_dataset(fold_train),"test":make_dataset(fold_test),"covered_notes":sorted(list(covered))}
        with open(f"dataset/folds/fold{i}.json","w") as f:
            json.dump(result,f)

def make_full_coverage_triple_folds():
    best_cross_validation_folds = None
    for attempt in range(500):
        folds = []
        all_covered = set()
        missing = graph.utils.missing_notes(all_covered)
        best = len(missing)
        all_seen = set()
        while len(all_seen) < 68:
            trn_edges, tst_edges, vld_edges = random_split_triple_carving(.5,.25,.25)
            all_covered = get_covered_notes(trn_edges).intersection(get_covered_notes(tst_edges)).intersection(get_covered_notes(vld_edges))
            missing = graph.utils.missing_notes(all_covered)
            best = min(best,len(missing))
            all_seen.update(all_covered)
            # print(i,"Current",len(missing),"Best",best,"All Seen",len(all_seen))
            folds.append((trn_edges, tst_edges, vld_edges,all_covered))

        if not best_cross_validation_folds or len(folds) < len(best_cross_validation_folds):
            best_cross_validation_folds = folds

        print(f"Finished attempt {attempt} after {len(folds)} folds. Best is {len(best_cross_validation_folds)}")

    for i, (fold_train, fold_test, fold_validate, fold_all_covered) in enumerate(best_cross_validation_folds):
        # "covered_notes" only has notes that appear in all datasets.
        # There may be notes that appear only in the train or only in the test dataset
        # so they are not in covered notes. Also, it may not be the case that a datapoint has any notes in covered notes.
        result = {"train":make_dataset(fold_train),"test":make_dataset(fold_test),"validate":make_dataset(fold_validate),"covered_notes":sorted(list(fold_all_covered))}
        with open(f"dataset/folds/fold{i}.json","w") as f:
            json.dump(result,f)

def find_single_triple_fold():
    all_covered = set()
    missing = graph.utils.missing_notes(all_covered)
    i = 0
    best = len(missing)
    all_seen = set()
    while missing:
        trn_edges, tst_edges, vld_edges = random_split_triple_carving(.5,.25,.25)
        all_covered = get_covered_notes(trn_edges).intersection(get_covered_notes(tst_edges)).intersection(get_covered_notes(vld_edges))
        missing = graph.utils.missing_notes(all_covered)
        best = max(best,len(missing))
        all_seen.update(all_covered)
        print(i,"Current",len(missing),"Best",best,"All Seen",len(all_seen))
        i+=1

    result = {"train":make_dataset(trn_edges),"test":make_dataset(tst_edges),"validate":make_dataset(vld_edges),"covered_notes":sorted(list(all_covered))}
    with open(f"dataset/triple_fold.json","w") as f:
        json.dump(result,f)

def attempt_full_coverage():
    all_covered = set()
    i = 0
    missing = graph.utils.missing_notes(all_covered)
    while i < 1000000:
        trn_edges, tst_edges = random_split_carving()
        covered = get_covered_notes(trn_edges).intersection(get_covered_notes(tst_edges))
        all_covered.update(covered)
        missing = graph.utils.missing_notes(all_covered)
        print(f"{i} = Covered: {len(all_covered)}. Missing ({len(missing)}): {missing}")
        i+=1

def anneal_better_coverage():
    def get_data(nodes):
        edges = build_edges(nodes)
        return edges, get_covered_notes(edges)

    all_covered = set()
    i = 0
    train_nodes = set(random.sample(sorted(all_nodes),int(len(all_nodes)*train_fraction)))
    test_nodes = all_nodes.difference(train_nodes)
    
    train_nodes = list(train_nodes)
    _, train_covered = get_data(train_nodes)

    test_nodes = list(test_nodes)
    _, test_covered = get_data(test_nodes)

    while i < 1000:
        x1,x2 = random.choice(train_nodes), random.choice(test_nodes)
        new_train_nodes = [n for n in train_nodes if n != x1] + [x2]
        _, new_train_covered = get_data(new_train_nodes)

        new_test_nodes = [n for n in test_nodes if n != x2] + [x1]
        _, new_test_covered = get_data(new_test_nodes)

        if len(new_train_covered) < len(train_covered):
            print("Skip",i,x1,x2)
            continue

        if len(new_test_covered) < len(test_covered):
            print("Skip",i,x1,x2)
            continue

        train_nodes, train_covered = new_train_nodes, new_train_covered
        test_nodes, test_covered = new_test_nodes, new_test_covered
        print("Swap",i,x1,x2,len(train_covered),len(test_covered))
        i += 1



# make_full_coverage_triple_folds()
anneal_better_coverage()