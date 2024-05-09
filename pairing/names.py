import json
import pairing.data

pairings, all_smiles, note_counts, name_to_smiles = pairing.data.get_pairings()
with open("charts/data/name_to_smiles.json",'w') as f:
    json.dump(name_to_smiles,f)