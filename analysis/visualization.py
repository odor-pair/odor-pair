import matplotlib.pyplot as plt
import networkx as nx
from pairing.data import PairData, Dataset, loader
import pairing.data
import random
import tqdm
import collections
import re

def smiles_list(is_train):
    smiles = set()
    counter = collections.Counter()
    ds = pairing.data.Dataset(is_train=is_train)
    for d in tqdm.tqdm(ds):
        smiles.add(d.smiles_s)
        counter[d.smiles_s] += 1

        smiles.add(d.smiles_t)
        counter[d.smiles_t] += 1

    return list(smiles), counter

fraction = .01

train_smiles,train_counter = smiles_list(True)
test_smiles,test_counter = smiles_list(False)
print(len(train_smiles),len(test_smiles))

def sample_fraction(smiles):
    global fraction
    return set(random.sample(smiles,int(fraction*len(smiles))))

train_smiles = set([k for k,_ in train_counter.most_common(int(fraction*len(train_counter)))])
test_smiles = set([k for k,_ in test_counter.most_common(int(fraction*len(test_counter)))])

smiles_sample = train_smiles.union(test_smiles)
print(len(train_smiles),len(test_smiles),len(smiles_sample))


pairings, all_smiles, note_counts, name_to_smiles = pairing.data.get_pairings()
smiles_to_name = {v:k for k,v in name_to_smiles.items()}

G = nx.Graph()
for pair in pairings:
    m1,m2 = pair
    if not m1 in smiles_sample or not m2 in smiles_sample:
        continue
    G.add_edge(m1,m2)

def get_edge_style(edge):
    m1,m2 = edge
    if m1 in train_smiles and m2 in train_smiles:
        return 'dashed', 'lightsteelblue', .5
    if m1 in train_smiles and m2 in test_smiles:
        return 'dashed', 'lightsalmon', .5
    return 'solid', 'grey', 1

train_color = "blue"
test_color = "red"

plt.figure(3,figsize=(8,8)) 
color_map = [train_color if node in train_smiles else test_color for node in G]     
edge_styles, edge_colors, edge_weights = list(zip(*[get_edge_style(edge) for edge in G.edges()]))
print(collections.Counter(edge_styles))
print(G.number_of_nodes())
pos = nx.fruchterman_reingold_layout(G)
node_size = 40


def clean(name):
    # Remove parentheticals
    return re.sub(r"[\(\[].*?[\)\]]", "", name)

smiles_to_name = {k:clean(v) for k,v in smiles_to_name.items() if k in pos}

nx.draw_networkx_nodes(G, pos, node_size=node_size, nodelist=train_smiles,
                       node_color=train_color, label='Train Molecules')
nx.draw_networkx_nodes(G, pos, node_size=node_size, nodelist=test_smiles,
                       node_color=test_color, label='Test Molecules')


nx.draw_networkx_edges(G, pos,style=edge_styles,edge_color=edge_colors,width=edge_weights)

labels_pos = {k:[x,y+3e-2] for k,(x,y) in pos.items()}
nx.draw_networkx_labels(G, labels_pos,font_weight="heavy",labels=smiles_to_name,font_size=8)

plt.legend()

plt.box(False)
carving_width = collections.Counter(edge_styles)["solid"]
print(f"{len(train_smiles)} train molecules and {len(test_smiles)} test molecules.")
print(f"Average degree = {int(G.number_of_edges()/G.number_of_nodes())} and carving width = {carving_width}")
plt.title("Molecule pairing meta-graph",fontweight="bold")

plt.tight_layout()
plt.show()
