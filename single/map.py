import single.data
import matplotlib.pyplot as plt
import sklearn
import sklearn.decomposition
import torch
import numpy as np
import pairing.data

all_notes = np.array(pairing.data.get_all_notes())
all_notes_lst = all_notes.tolist()

def get_notes(y):
    idcs = y.nonzero()
    return all_notes[idcs].tolist()

def get_color(notes):
    if len(notes) > 1:
        return 0
    n = notes[0]
    return all_notes_lst.index(n)

def make_chart(all_embed,all_y):
    data_notes = [get_notes(y) for y in all_y]
    all_c = [get_color(notes) for notes in data_notes]

    all_embed = [e for i,e in enumerate(all_embed) if all_c[i] !=0]
    all_y = [y for i,y in enumerate(all_y) if all_c[i] !=0]
    colors = [c for c in all_c if c !=0]
    pca = sklearn.decomposition.PCA(n_components=2)
    coords = pca.fit_transform(all_embed)

    print(len(colors))
    plt.scatter(coords[:,0],coords[:,1],c=colors)
    plt.show()
