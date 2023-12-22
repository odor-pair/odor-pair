import pairing.data

import collections
import numpy as np

import matplotlib.pyplot as plt


zipfs = collections.Counter()
occurs = collections.Counter()

pairings, _, _, _ = pairing.data.get_pairings()
for (m1,m2), notes in pairings.items():
    for i, note in enumerate(notes.keys()):
        zipfs[note] += 1/(i+1)
        occurs[note] += 1

notes = np.array([n for n, _ in occurs.most_common()])
zipfs_avg = np.array([zipfs[n]/f for n, f in occurs.most_common()])

idcs = np.argsort(zipfs_avg)

notes = notes[idcs]
zipfs_avg = zipfs_avg[idcs]

lim = 25

idxs = [i for i in range(len(notes))]

f, axs = plt.subplots(2, 1, figsize=(15, 6))
print(np.min(zipfs_avg),np.max(zipfs_avg),len(notes))


print(list(zip(notes,zipfs_avg)))

axs[0].bar(idxs[:lim],list(reversed(zipfs_avg))[:lim],align='edge')
axs[0].set_xticks(ticks=idxs[:lim],labels=list(reversed(notes))[:lim],rotation=45)

axs[1].bar(idxs[:lim],zipfs_avg[:lim],align='edge')
axs[1].set_xticks(ticks=idxs[:lim],labels=notes[:lim],rotation=45)
plt.tight_layout()
plt.show()