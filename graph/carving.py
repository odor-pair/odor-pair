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

for n1, n2 in edges:
    assert not (n2,n1) in edges
    assert not n1 == n2

full_graph = collections.defaultdict(set)
for n1, n2 in edges:
    full_graph[n1].add(n2)
    full_graph[n2].add(n1)

train_fraction = .85
test_fraction = .15

assert train_fraction + test_fraction == 1

class FrontierGraph(object):
    def __init__(self, start):
        self.nodes = {start}
        self.frontier = set(full_graph[start])

    def can_expand(self,forbidden):
        possible = self.frontier.difference(forbidden)
        return len(possible) > 0

    def expand(self,forbidden):
        # We could update this in place instead to reduce complexity
        possible = self.frontier.difference(forbidden)
        if not possible:
            return

        next_node = random.choice(sorted(possible))
        
        assert not next_node in self.nodes
        self.nodes.add(next_node)
        self.frontier.remove(next_node)
        
        unseen = full_graph[next_node].difference(self.nodes)
        self.frontier.update(unseen)

        assert not self.nodes.intersection(self.frontier)

    def build_edges(self):
        all_edges = set()
        for node in self.nodes:
            for other in full_graph[node]:
                if not other in self.nodes:
                    continue
                all_edges.add(sort(node,other))
        return all_edges



def dfs_carving():
    train_start, test_start = random.sample(sorted(nodes),2)

    train_graph = FrontierGraph(train_start)
    test_graph = FrontierGraph(test_start)
    train_possible = True
    test_possible = True

    while train_possible or test_possible:
        if not test_possible:
            train_possible = train_graph.expand(test_graph.nodes)
            continue
        if not train_possible:
            test_possible = test_graph.expand(train_graph.nodes)
            continue

        if random.random() < train_fraction:
            train_graph.expand(test_graph.nodes)
        else:
            test_graph.expand(train_graph.nodes)

        train_possible = train_graph.can_expand(test_graph.nodes)
        test_possible = test_graph.can_expand(train_graph.nodes)

    train_edges = train_graph.build_edges()
    test_edges = test_graph.build_edges()

    assert not train_graph.nodes.intersection(test_graph.nodes)
    assert not train_edges.intersection(test_edges)
    
    return train_edges, test_edges



best = 0
best_trn, best_tst = {}, {}
for i in tqdm.tqdm(range(5000)):
    trn_edges, tst_edges = dfs_carving()
    total = len(trn_edges) + len(tst_edges)
    score = graph.stats.count_nonzero(trn_edges) + graph.stats.count_nonzero(tst_edges)
    print(f"Made {total:,.0f} w/ score={score:,.0f}")

    if score > best:
        best = score
        best_trn, best_tst = trn_edges, tst_edges
    print(f"Best = {best:,.0f} w/ {len(best_trn)} and {len(best_tst)}. (Missing {2*len(graph.utils.CANON_NOTES_LIST)-best})")

trn_sim = graph.stats.kl_similarity(best_trn)
tst_sim = graph.stats.kl_similarity(best_tst)
print(f"From a total of {len(edges):,} edges, we carved {len(best_trn)+len(best_tst):,} edges ({len(best_trn):,} train edges w/ sim={trn_sim:.2f} + {len(best_tst):,} test edges w/ sim={tst_sim:.2f}), losing {len(edges)-best:,} edges.")
result = {"train":[list(e) for e in best_trn],"test":[list(e) for e in best_tst]}
with open("dataset/carving.json","w") as f:
    json.dump(result,f)