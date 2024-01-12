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

def compare_models(use_multiple_folds=False):
    start = time.perf_counter()
    if use_multiple_folds:
        fold_datasets = graph.folds.load_fold_datasets()
    else:
        fold_datasets = graph.folds.load_single_fold()
    print(f"Loaded dataset in {time.perf_counter()-start:.1f}")
    note_to_score = {'acidic': 0.94404584, 'alcoholic': 0.93257785, 'aldehydic': 0.81184626, 'alliaceous': 0.70352423, 'amber': 0.8645878, 'animal': 0.85070544, 'anise': 0.85907114, 'aromatic': 0.39118046, 'balsamic': 0.78261554, 'berry': 0.8071169, 'bitter': 0.5778024, 'bready': 0.30419004, 'brown': 0.53877926, 'burnt': 0.3027718, 'buttery': 0.82760674, 'camphoreous': 0.72009975, 'caramellic': 0.6871636, 'cheesy': 0.98356014, 'chemical': 0.07104282, 'chocolate': 0.9734151, 'citrus': 0.6302929, 'clean': 0.6637347, 'cocoa': 0.29191083, 'coconut': 0.9104239, 'coffee': 0.8831153, 'cooling': 0.58075666, 'coumarinic': 0.37027624, 'creamy': 0.56295085, 'dairy': 0.12101512, 'earthy': 0.43979374, 'estery': 0.37520128, 'ethereal': 0.61340153, 'fatty': 0.5691519, 'fermented': 0.6206701, 'floral': 0.6764056, 'fresh': 0.92440003, 'fruity': 0.6861677, 'fungal': 0.3814531, 'fusel': 0.48335287, 'green': 0.656567, 'herbal': 0.6152357, 'honey': 0.8776704, 'jammy': 0.4876382, 'licorice': 0.7828236, 'marine': 0.44851583, 'meaty': 0.6492507, 'medicinal': 0.6637752, 'melon': 0.850806, 'mentholic': 0.62330437, 'minty': 0.71491694, 'mossy': 0.93572277, 'musk': 0.80527997, 'mustard': 0.42574495, 'musty': 0.48075286, 'nutty': 0.5825912, 'oily': 0.7669942, 'onion': 0.5315418, 'orris': 0.44660679, 'phenolic': 0.5361419, 'pine': 0.22661729, 'powdery': 0.5620614, 'roasted': 0.5846187, 'rummy': 0.4662616, 'soapy': 0.47403157, 'solvent': 0.57658935, 'sour': 0.80184096, 'spicy': 0.6675265, 'sulfurous': 0.8201014, 'sweet': 0.65269506, 'thujonic': 0.97590184, 'tonka': 0.38637573, 'tropical': 0.45148665, 'vanilla': 0.96550703, 'vegetable': 0.5568697, 'waxy': 0.611503, 'winey': 0.6989641, 'woody': 0.7555574}
    # note_to_score = get_baseline_score(fold_datasets)
    print(note_to_score)
    # If we go back to using folds, we will have to change the way we calculate
    # low data notes,
    analysis.auroc.make_chart_from_dictionary(note_to_score,fold_datasets[0]["train_frequencies"],fold_datasets[0]["test_frequencies"])

compare_models()
# nts = 
# analysis.auroc.make_chart_from_dictionary(nts)

