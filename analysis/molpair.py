import torch
import rdkit.Chem.Draw
from rdkit import Chem
from pairing.data import PairData, Dataset
import pairing.data
import matplotlib.pyplot as plt
import numpy as np

test = Dataset(is_train=True)
all_notes = np.array(pairing.data.get_all_notes())

def get_notes(d):
    idcs = d.y.nonzero().squeeze().numpy()
    return all_notes[idcs]

def find_exciting_pair():
    for i,d in enumerate(test):
        if len(get_notes(d)) > 4:
            return i

def format(name,notes):
    notes = [f"\"{n}\"" for n in notes]
    # Matplotlib is annoying and combines words used in a single bf mathtext.
    # Split, bold, then rejoin.
    w = [r'$\bf{' + w + '}$' for w in name.split()]
    name = " ".join(w)
    return f"{name}\n{', '.join(notes)}"

# Selected because it has many labels
d = test[13386]

name1 = "Cyclohexyl formate"
mol1 = Chem.MolFromSmiles(d.smiles_s)
notes1 = ['ethereal', 'fruity', 'cooling', 'green']
im1 = Chem.Draw.MolToImage(mol1)

name2 = "Methyl 2-hexenoate"
mol2 = Chem.MolFromSmiles(d.smiles_t)
notes2 = ['herbal', 'fruity']
im2 = Chem.Draw.MolToImage(mol2)


# plt.rc('text', usetex=True)
fig, axarr = plt.subplots(1,2)
axarr[0].imshow(im1)
axarr[0].set_title(format(name1,notes1))
axarr[1].imshow(im2)
axarr[1].set_title(format(name2,notes2))
# Remove ticks but keep boxes around molecules.
for ax in axarr:
    ax.set_xticks([])
    ax.set_yticks([])

print(get_notes(d))
fig.suptitle(format("Blend contains notes distinct from either molecule",get_notes(d)))
plt.show()


