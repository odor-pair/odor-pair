import time
import graph.folds
import analysis.fingerprint
import pairing.comparison
import analysis.auroc
import collections
import numpy as np


def get_baseline_score(fold_datasets):
    notes_to_scores = collections.defaultdict(list)
    for i,dataset in enumerate(fold_datasets):
        start = time.perf_counter()
        mfpgen = analysis.fingerprint.make_mfpgen()

        train_embed, train_y = pairing.comparison.collate(mfpgen,dataset["train"])
        test_embed, test_y = pairing.comparison.collate(mfpgen,dataset["test"])
        pred, y = analysis.fingerprint.get_test_pred_y(train_embed, train_y, test_embed, test_y)
        for note, score in analysis.auroc.make_score_dict(pred,y,dataset["covered_notes"]).items():
            notes_to_scores[note].append(score)
        
        print(f"Trained model for fold {i} in {time.perf_counter()-start:.1f}")

    return {note:np.mean(scores) for note, scores in notes_to_scores.items()}

def compare_models():
    start = time.perf_counter()
    fold_datasets = graph.folds.load_fold_datasets()
    print(f"Loaded dataset in {time.perf_counter()-start:.1f}")
    note_to_score = get_baseline_score(fold_datasets)
    print(note_to_score)
    analysis.auroc.make_chart_from_dictionary(note_to_score)

# compare_models()
nts = {'acidic': 0.9345935, 'alcoholic': 0.8206077, 'aldehydic': 0.80087095, 'alliaceous': 0.60822153, 'amber': 0.7830501, 'animal': 0.8824658, 'anise': 0.8494186, 'aromatic': 0.44208512, 'balsamic': 0.7311323, 'berry': 0.7812042, 'bitter': 0.38841033, 'bready': 0.294824, 'brown': 0.5368073, 'burnt': 0.62852734, 'buttery': 0.8179924, 'camphoreous': 0.6082048, 'caramellic': 0.70113546, 'cheesy': 0.9735174, 'chemical': 0.5222834, 'chocolate': 0.70166534, 'citrus': 0.6151182, 'clean': 0.6141081, 'cocoa': 0.48996493, 'coconut': 0.61591697, 'coffee': 0.8428262, 'cooling': 0.46033087, 'coumarinic': 0.5808179, 'creamy': 0.6434711, 'earthy': 0.5137241, 'estery': 0.5673819, 'ethereal': 0.4907577, 'fatty': 0.6381226, 'fermented': 0.49508694, 'floral': 0.65329367, 'fresh': 0.85226965, 'fruity': 0.66087127, 'fungal': 0.28057453, 'fusel': 0.6428997, 'green': 0.5817153, 'herbal': 0.55976295, 'honey': 0.782136, 'licorice': 0.74124706, 'marine': 0.69823045, 'meaty': 0.8392119, 'medicinal': 0.5252871, 'melon': 0.71838874, 'mentholic': 0.8375193, 'minty': 0.5980165, 'mossy': 0.9480268, 'musk': 0.9102171, 'musty': 0.5083849, 'nutty': 0.68002397, 'oily': 0.6327002, 'onion': 0.75001746, 'orris': 0.39034912, 'phenolic': 0.68468064, 'powdery': 0.8427431, 'roasted': 0.58141506, 'rummy': 0.38243127, 'soapy': 0.58446413, 'sour': 0.7281835, 'spicy': 0.64259917, 'sulfurous': 0.7760914, 'sweet': 0.46166328, 'thujonic': 0.95023966, 'tonka': 0.3652207, 'tropical': 0.5347624, 'vanilla': 0.9492879, 'vegetable': 0.63725525, 'waxy': 0.67733717, 'winey': 0.55040175, 'woody': 0.6973284, 'jammy': 0.17729035, 'mustard': 0.8279638, 'pine': 0.37027118, 'solvent': 0.39667854, 'dairy': 0.35186687}
analysis.auroc.make_chart_from_dictionary(nts)

