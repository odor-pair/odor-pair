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

def get_best_model_score(fold_datasets):
    notes_to_scores = collections.defaultdict(list)
    for i,dataset in enumerate(fold_datasets):
        pred, y = analysis.best.collate_test(dataset)
        for note, score in analysis.auroc.make_score_dict(pred,y,dataset["covered_notes"]).items():
            notes_to_scores[note].append(score)

    return {note:np.mean(scores) for note, scores in notes_to_scores.items()}

def compare_models(use_multiple_folds=False,remake_baseline=False):
    start = time.perf_counter()
    if use_multiple_folds:
        fold_datasets = graph.folds.load_fold_datasets()
    else:
        fold_datasets = graph.folds.load_single_fold()
    print(f"Loaded dataset in {time.perf_counter()-start:.1f}")

    if remake_baseline:
        baseline_note_to_score = get_baseline_score(fold_datasets)
    else:
        baseline_note_to_score = {'acidic': 0.94404584, 'alcoholic': 0.93257785, 'aldehydic': 0.81184626, 'alliaceous': 0.70352423, 'amber': 0.8645878, 'animal': 0.85070544, 'anise': 0.85907114, 'aromatic': 0.39118046, 'balsamic': 0.78261554, 'berry': 0.8071169, 'bitter': 0.5778024, 'bready': 0.30419004, 'brown': 0.53877926, 'burnt': 0.3027718, 'buttery': 0.82760674, 'camphoreous': 0.72009975, 'caramellic': 0.6871636, 'cheesy': 0.98356014, 'chemical': 0.07104282, 'chocolate': 0.9734151, 'citrus': 0.6302929, 'clean': 0.6637347, 'cocoa': 0.29191083, 'coconut': 0.9104239, 'coffee': 0.8831153, 'cooling': 0.58075666, 'coumarinic': 0.37027624, 'creamy': 0.56295085, 'dairy': 0.12101512, 'earthy': 0.43979374, 'estery': 0.37520128, 'ethereal': 0.61340153, 'fatty': 0.5691519, 'fermented': 0.6206701, 'floral': 0.6764056, 'fresh': 0.92440003, 'fruity': 0.6861677, 'fungal': 0.3814531, 'fusel': 0.48335287, 'green': 0.656567, 'herbal': 0.6152357, 'honey': 0.8776704, 'jammy': 0.4876382, 'licorice': 0.7828236, 'marine': 0.44851583, 'meaty': 0.6492507, 'medicinal': 0.6637752, 'melon': 0.850806, 'mentholic': 0.62330437, 'minty': 0.71491694, 'mossy': 0.93572277, 'musk': 0.80527997, 'mustard': 0.42574495, 'musty': 0.48075286, 'nutty': 0.5825912, 'oily': 0.7669942, 'onion': 0.5315418, 'orris': 0.44660679, 'phenolic': 0.5361419, 'pine': 0.22661729, 'powdery': 0.5620614, 'roasted': 0.5846187, 'rummy': 0.4662616, 'soapy': 0.47403157, 'solvent': 0.57658935, 'sour': 0.80184096, 'spicy': 0.6675265, 'sulfurous': 0.8201014, 'sweet': 0.65269506, 'thujonic': 0.97590184, 'tonka': 0.38637573, 'tropical': 0.45148665, 'vanilla': 0.96550703, 'vegetable': 0.5568697, 'waxy': 0.611503, 'winey': 0.6989641, 'woody': 0.7555574}

    openpom = {"acidic":0.982218413,"aldehydic":0.9060993115,"alliaceous":0.8939357183,"amber":0.9456035004,"animal":0.592631784,"anise":0.8507702956,"aromatic":0.781771179,"balsamic":0.8912972094,"berry":0.9091650146,"bitter":0.4516077492,"bready":0.7402336337,"brown":0.8422859267,"burnt":0.818552605,"buttery":0.9456593649,"camphoreous":0.851641635,"caramellic":0.8750726348,"cheesy":0.951501981,"chemical":0.7326678619,"chocolate":0.9477555388,"citrus":0.8031868172,"clean":0.8495126217,"cocoa":0.5680993424,"coconut":0.9052183871,"coffee":0.9402027227,"cooling":0.3547693724,"coumarinic":0.8185648704,"creamy":0.7778984577,"dairy":0.5596644937,"earthy":0.3693028324,"estery":0.7046878568,"ethereal":0.7962966713,"fatty":0.7005376413,"fermented":0.7509624803,"floral":0.7763464445,"fresh":0.9104471927,"fruity":0.7834931395,"fungal":0.7430793668,"fusel":0.6949714063,"green":0.6359466235,"herbal":0.6636192617,"honey":0.9235785051,"jammy":0.8275073313,"licorice":0.8562556453,"marine":0.5754463587,"meaty":0.917129558,"medicinal":0.8546474245,"melon":0.4045912577,"mentholic":0.7582853976,"minty":0.734904461,"mossy":0.9407550716,"musk":0.9133981961,"musty":0.6195661785,"nutty":0.8056712112,"oily":0.5514932078,"onion":0.9526747059,"orris":0.290991969,"phenolic":0.7457307943,"powdery":0.6347801917,"roasted":0.8487581538,"rummy":0.5577875458,"soapy":0.6685724308,"solvent":0.7238740741,"sour":0.987147486,"spicy":0.7332137062,"sulfurous":0.8197665246,"sweet":0.5768583456,"thujonic":0.5342897452,"tonka":0.8232654476,"tropical":0.5217778292,"vanilla":0.9758381567,"vegetable":0.8378683019,"waxy":0.81878502,"winey":0.6864727652,"woody":0.8692144586}
    ensemble_openpom = {"acidic":0.9947609001,"aldehydic":0.9439193869,"alliaceous":0.9287700773,"amber":0.9598485651,"animal":0.6419004976,"anise":0.8766194359,"aromatic":0.8107905355,"balsamic":0.9161091967,"berry":0.9359506657,"bitter":0.4198871134,"bready":0.8285035629,"brown":0.8994965098,"burnt":0.8977012018,"buttery":0.9682076996,"camphoreous":0.880528026,"caramellic":0.9129761651,"cheesy":0.9744257045,"chemical":0.7792486205,"chocolate":0.9704135211,"citrus":0.8361816449,"clean":0.9257989664,"cocoa":0.6058186839,"coconut":0.9529566711,"coffee":0.9602257676,"cooling":0.3308365067,"coumarinic":0.8960198839,"creamy":0.8318599775,"dairy":0.5776346768,"earthy":0.3530386887,"estery":0.7345436552,"ethereal":0.8519862688,"fatty":0.7087429437,"fermented":0.809978525,"floral":0.7927365177,"fresh":0.9729334587,"fruity":0.7996205013,"fungal":0.825926219,"fusel":0.7428976938,"green":0.651153853,"herbal":0.6966326625,"honey":0.9518874552,"jammy":0.8915067651,"licorice":0.8974784777,"marine":0.5978735979,"meaty":0.9574152318,"medicinal":0.8713096231,"melon":0.3863254897,"mentholic":0.8579197102,"minty":0.7671224251,"mossy":0.9848510164,"musk":0.9326614806,"musty":0.6374486241,"nutty":0.8612899061,"oily":0.6070318419,"onion":0.9680462871,"orris":0.2190662376,"phenolic":0.7921459293,"powdery":0.6631794411,"roasted":0.903344803,"rummy":0.5905407903,"soapy":0.7050296717,"solvent":0.8291284576,"sour":0.997771977,"spicy":0.7788769697,"sulfurous":0.8478858492,"sweet":0.5912518456,"thujonic":0.5515635117,"tonka":0.8902711303,"tropical":0.5314903704,"vanilla":0.9871341614,"vegetable":0.8751406839,"waxy":0.8362613615,"winey":0.7055465218,"woody":0.8930969902}
    best_note_to_score = get_best_model_score(fold_datasets)
    # If we go back to using folds, we will have to change the way we calculate
    # low data notes,
    all_note_to_scores = [best_note_to_score,baseline_note_to_score,openpom,ensemble_openpom]
    all_model_names = ["Best Model","MFP Baseline","OpenPOM", "EnsembleOpenPOM"]
    analysis.auroc.make_chart_from_dictionary(all_note_to_scores,all_model_names,fold_datasets[0]["train_frequencies"],fold_datasets[0]["test_frequencies"])

compare_models()
# nts = 
# analysis.auroc.make_chart_from_dictionary(nts)

