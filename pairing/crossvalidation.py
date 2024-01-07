import time
import graph.folds
import analysis.fingerprint
import pairing.comparison
import analysis.auroc
import collections
import numpy as np


def get_baseline_score(fold_datasets):
    notes_to_scores = collections.defaultdict(list)
    for dataset in fold_datasets:
        mfpgen = analysis.fingerprint.make_mfpgen()

        train_embed, train_y = pairing.comparison.collate(mfpgen,dataset["train"][:1000])
        test_embed, test_y = pairing.comparison.collate(mfpgen,dataset["test"][:1000])
        pred, y = analysis.fingerprint.get_test_pred_y(train_embed, train_y, test_embed, test_y)
        for note, score in analysis.auroc.make_score_dict(pred,y,dataset["covered_notes"]).items():
            notes_to_scores[note].append(score)

    fingerprint_scores = {note:np.mean(scores) for note, scores in notes_to_scores.items()}
    print(fingerprint_scores)

def compare_models():
    start = time.perf_counter()
    fold_datasets = graph.folds.load_fold_datasets()
    print(f"Loaded dataset in {time.perf_counter()-start:.0f}")

nts = {'acidic': 0.32030729452768963, 'alcoholic': 0.0, 'aldehydic': 0.9093894561131796, 'alliaceous': 0.7484699885050455, 'amber': 0.86565234263738, 'animal': 0.5242947439352671, 'anise': 0.8256589968999227, 'aromatic': 0.0, 'balsamic': 0.8726365367571512, 'berry': 0.9079002936681112, 'bitter': 0.46084335446357727, 'bready': 0.5, 'brown': 0.7552940050760905, 'burnt': 0.8442046046257019, 'buttery': 0.8855204780896505, 'camphoreous': 0.7728910048802694, 'caramellic': 0.9253307779630026, 'cheesy': 0.4969879388809204, 'chemical': 0.607383112112681, 'chocolate': 0.8464348514874777, 'citrus': 0.6239181955655416, 'clean': 0.6358024676640829, 'cocoa': 0.3333333333333333, 'coconut': 0.9603581229845682, 'coffee': 0.8760972817738851, 'cooling': 0.5415247281392416, 'coumarinic': 0.9750570058822632, 'creamy': 0.7619105577468872, 'earthy': 0.5636534492174784, 'estery': 0.4973155657450358, 'ethereal': 0.7631781498591105, 'fatty': 0.763046145439148, 'fermented': 0.46942296624183655, 'floral': 0.7620025873184204, 'fresh': 0.25, 'fruity': 0.774093230565389, 'fungal': 0.6546546816825867, 'fusel': 0.7040245234966278, 'green': 0.6026460925738016, 'herbal': 0.6456391215324402, 'honey': 0.8807871540387472, 'licorice': 0.0, 'marine': 0.25, 'meaty': 0.6056998570760092, 'medicinal': 0.25, 'melon': 0.33059905966122943, 'mentholic': 0.0, 'minty': 0.7817433476448059, 'mossy': 0.25, 'musk': 0.9167123635609945, 'musty': 0.5955398778120676, 'nutty': 0.7541918953259786, 'oily': 0.44201306502024335, 'onion': 0.7211452921231588, 'orris': 0.5, 'phenolic': 0.4248777429262797, 'powdery': 0.7319819927215576, 'roasted': 0.35920705397923786, 'rummy': 0.544107973575592, 'soapy': 0.7876623670260111, 'sour': 0.0, 'spicy': 0.7217118541399637, 'sulfurous': 0.9204627871513367, 'sweet': 0.0, 'thujonic': 0.35621243715286255, 'tonka': 0.38792669773101807, 'tropical': 0.42463480432828266, 'vanilla': 0.9019820888837179, 'vegetable': 0.6937973499298096, 'waxy': 0.7181431849797567, 'winey': 0.5768506228923798, 'woody': 0.798441211382548, 'jammy': 0.23323321342468262, 'mustard': 0.0, 'pine': 0.0, 'solvent': 0.0, 'dairy': 0.8324975371360779}
analysis.auroc.make_chart_from_dictionary(nts)

