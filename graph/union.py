import json
import torch
import tqdm
import torchmetrics

import numpy as np
import analysis.fingerprint
import sklearn
import sklearn.model_selection
import warnings

with open("dataset/full.json") as f:
    full_data = json.load(f)

all_blend_notes = set()
all_single_notes = set()
for d in full_data:
    all_blend_notes.update(d["blend_notes"])
    all_single_notes.update(d["mol1_notes"])
    all_single_notes.update(d["mol2_notes"])

def canonize(notes):
    canon = {"":None,"No odor group found for these":None,
    "anisic":"anise","corn chip":"corn",
    "medicinal,":"medicinal"}

    cleaned = set()
    for n in notes:
        # Valid note
        if not n in canon:
            cleaned.add(n)
        # Should be removed
        elif not canon[n]:
            continue
        else:
            cleaned.add(canon[n])
    return cleaned

# Conver to list so indexing is faster.
all_blend_notes = list(canonize(all_blend_notes))
print(sorted(all_blend_notes))
all_single_notes = list(canonize(all_single_notes))

canon_dict = {'absinthe': 'alcoholic',  'acacia': 'coconut',  'acetic': 'acetic',  'acetone': 'acetic',  'acidic': 'acidic',  'acorn': 'corn',  'acrylate': 'phenolic',  'agarwood': 'woody',  'alcoholic': 'alcoholic',  'aldehydic': 'aldehydic',  'algae': 'fungal',  'alliaceous': 'alliaceous',  'allspice': 'alliaceous',  'almond': 'nutty',  'almond bitter almond': 'bitter',  'almond roasted almond': 'roasted',  'almond toasted almond': 'toasted',  'amber': 'amber',  'ambergris': 'amber',  'ambrette': 'amber',  'ammoniacal': 'ammoniacal',  'angelica': 'anise',  'animal': 'animal',  'anise': 'anise',  'anisic': 'anise',  'apple': 'fruity',  'apple cooked apple': 'roasted',  'apple dried apple': 'fruity',  'apple green apple': 'green',  'apple skin': 'melon',  'apricot': 'fruity',  'aromatic': 'aromatic',  'arrack': 'aromatic',  'artichoke': 'celery',  'asparagus': 'vegetable',  'astringent': 'pungent',  'autumn': 'floral',  'bacon': 'meaty',  'baked': 'bready',  'balsamic': 'balsamic',  'banana': 'melon',  'banana peel': 'melon',  'banana ripe banana': 'fruity',  'banana unripe banana': 'melon',  'barley roasted barley': 'roasted',  'basil': 'herbal',  'bay': 'marine',  'bean green bean': 'vegetable',  'beany': 'phenolic',  'beef juice': 'meaty',  'beefy': 'meaty',  'beefy roasted beefy': 'meaty',  'beeswax': 'waxy',  'benzoin': 'phenolic',  'bergamot': 'minty',  'berry': 'berry',  'berry ripe berry': 'berry',  'bitter': 'bitter',  'blackberry': 'berry',  'bloody': 'pungent',  'blueberry': 'berry',  'bois de rose': 'floral',  'boronia': 'floral',  'bouillon': 'onion',  'brandy': 'alcoholic',  'bread baked': 'bready',  'bread crust': 'bready',  'bread rye bread': 'bready',  'bready': 'bready',  'broccoli': 'cabbage',  'brothy': 'bready',  'brown': 'brown',  'bubble gum': 'minty',  'buchu': 'coumarinic',  'burnt': 'burnt',  'buttermilk': 'dairy',  'butterscotch': 'buttery',  'buttery': 'buttery',  'cabbage': 'cabbage',  'camphoreous': 'camphoreous',  'cananga': 'phenolic',  'candy': 'chocolate',  'cantaloupe': 'melon',  'capers': 'camphoreous',  'caramellic': 'caramellic',  'caraway': 'caramellic',  'cardamom': 'camphoreous',  'carnation': 'caramellic',  'carrot': 'vegetable',  'carrot seed': 'vegetable',  'carvone': 'caramellic',  'cashew': 'nutty',  'cassia': 'camphoreous',  'castoreum': 'camphoreous',  'catty': 'camphoreous',  'cauliflower': 'vegetable',  'cedar': 'woody',  'cedarwood': 'woody',  'celery': 'celery',  'cereal': 'dairy',  'chamomile': 'herbal',  'charred': 'burnt',  'cheesy': 'cheesy',  'cheesy bleu cheese': 'cheesy',  'cheesy feta cheese': 'cheesy',  'cheesy limburger cheese': 'cheesy',  'cheesy parmesan cheese': 'cheesy',  'cheesy roquefort cheese': 'cheesy',  'chemical': 'chemical',  'cherry': 'cherry',  'cherry maraschino cherry': 'cherry',  'chervil': 'cherry',  'chicken': 'eggy',  'chicken coup': 'eggy',  'chicken fat': 'fatty',  'chicken roasted chicken': 'roasted',  'chicory': 'celery',  'chocolate': 'chocolate',  'chocolate dark chocolate': 'chocolate',  'chrysanthemum': 'floral',  'cider': 'alcoholic',  'cilantro': 'spicy',  'cinnamon': 'caramellic',  'cinnamyl': 'caramellic',  'cistus': 'citrus',  'citronella': 'citrus',  'citrus': 'citrus',  'citrus peel': 'citrus',  'citrus rind': 'citrus',  'civet': 'coumarinic',  'clam': 'camphoreous',  'clean': 'clean',  'cloth laundered cloth': 'leathery',  'clove': 'herbal',  'clover': 'floral',  'cocoa': 'cocoa',  'coconut': 'coconut',  'coffee': 'coffee',  'coffee roasted coffee': 'coffee',  'cognac': 'winey',  'cologne': 'musk',  'cooked': 'roasted',  'cookie': 'bready',  'cooling': 'cooling',  'coriander': 'herbal',  'corn': 'corn',  'corn chip': 'corn',  'cornmint': 'minty',  'cortex': 'phenolic',  'costus': 'coumarinic',  'cotton candy': 'cocoa',  'coumarinic': 'coumarinic',  'cranberry': 'berry',  'creamy': 'creamy',  'cucumber': 'cabbage',  'cucumber skin': 'cabbage',  'cumin': 'herbal',  'currant black currant': 'berry',  'currant bud black currant bud': 'herbal',  'custard': 'creamy',  'cyclamen': 'floral',  'cypress': 'citrus',  'dairy': 'dairy',  'date': 'phenolic',  'davana': 'vanilla',  'dewy': 'dusty',  'dill': 'garlic',  'dirty': 'dusty',  'dry': 'dusty',  'durian': 'fruity',  'dusty': 'dusty',  'earthy': 'earthy',  'egg nog': 'eggy',  'eggy': 'eggy',  'elderflower': 'floral',  'elemi': 'phenolic',  'estery': 'estery',  'ethereal': 'ethereal',  'eucalyptus': 'musk',  'fatty': 'fatty',  'fecal': 'brown',  'fennel': 'fusel',  'fenugreek': 'herbal',  'fermented': 'fermented',  'fig': 'phenolic',  'filbert': 'fusel',  'fir needle': 'pine',  'fishy': 'fishy',  'fleshy': 'musty',  'floral': 'floral',  'foliage': 'floral',  'forest': 'mossy',  'frankincense': 'musk',  'freesia': 'floral',  'fresh': 'fresh',  'fresh outdoors': 'fresh',  'fried': 'fatty',  'fruit dried fruit': 'fruity',  'fruit overripe fruit': 'fruity',  'fruit ripe fruit': 'fruity',  'fruit tropical fruit': 'fruity',  'fruity': 'fruity',  'fungal': 'fungal',  'fusel': 'fusel',  'galbanum': 'camphoreous',  'gardenia': 'floral',  'garlic': 'garlic',  'gasoline': 'musk',  'gassy': 'musty',  'genet': 'phenolic',  'geranium': 'floral',  'ginger': 'spicy',  'goaty': 'animal',  'gooseberry': 'berry',  'gourmand': 'tarragon',  'graham cracker': 'bready',  'grain': 'bready',  'grain toasted grain': 'toasted',  'grape': 'fruity',  'grape skin': 'fruity',  'grapefruit': 'fruity',  'grapefruit peel': 'fruity',  'grassy': 'earthy',  'gravy': 'meaty',  'greasy': 'oily',  'green': 'green',  'grilled': 'roasted',  'guaiacol': 'coumarinic',  'guaiacwood': 'woody',  'guava': 'fruity',  'hairy': 'waxy',  'ham': 'meaty',  'hawthorn': 'herbal',  'hay': 'hay',  'hay new mown hay': 'hay',  'hazelnut': 'nutty',  'hazelnut roasted hazelnut': 'roasted',  'heliotrope': 'phenolic',  'herbal': 'herbal',  'honey': 'honey',  'honeydew': 'honey',  'honeysuckle': 'honey',  'horseradish': 'hay',  'humus': 'herbal',  'hyacinth': 'herbal',  'immortelle': 'ethereal',  'incense': 'musk',  'jackfruit': 'fruity',  'jammy': 'jammy',  'jasmin': 'jammy',  'juicy': 'juicy',  'juicy fruit': 'fruity',  'juniper': 'pine',  'ketonic': 'acetic',  'kiwi': 'fruity',  'labdanum': 'lactonic',  'lactonic': 'lactonic',  'lamb': 'animal',  'lard': 'fatty',  'lavender': 'floral',  'leafy': 'floral',  'leathery': 'leathery',  'leek': 'lactonic',  'lemon': 'citrus',  'lemon peel': 'citrus',  'lemongrass': 'citrus',  'lettuce': 'cabbage',  'licorice': 'licorice',  'licorice black licorice': 'licorice',  'lilac': 'lactonic',  'lily': 'floral',  'lily of the valley': 'floral',  'lime': 'citrus',  'linden flower': 'floral',  'liver': 'medicinal',  'loganberry': 'berry',  'lovage': 'caramellic',  'lychee': 'cherry',  'mace': 'musk',  'magnolia': 'musk',  'mahogany': 'woody',  'malty': 'malty',  'mandarin': 'phenolic',  'mango': 'fruity',  'maple': 'pine',  'marine': 'marine',  'marjoram': 'herbal',  'marzipan': 'chocolate',  'meaty': 'meaty',  'meaty roasted meaty': 'meaty',  'medicinal': 'medicinal',  'melon': 'melon',  'melon rind': 'melon',  'mentholic': 'mentholic',  'metallic': 'chemical',  'milky': 'dairy',  'mimosa': 'fruity',  'minty': 'minty',  'molasses': 'honey',  'moldy': 'moldy',  'mossy': 'mossy',  'muguet': 'musk',  'mushroom': 'mushroom',  'musk': 'musk',  'mustard': 'mustard',  'musty': 'musty',  'mutton': 'meaty',  'myrrh': 'musk',  'naphthyl': 'phenolic',  'narcissus': 'phenolic',  'nasturtium': 'herbal',  'natural': 'herbal',  'neroli': 'floral',  'nut flesh': 'nutty',  'nut skin': 'nutty',  'nutmeg': 'nutty',  'nutty': 'nutty',  'oakmoss': 'mossy',  'oily': 'oily',  'onion': 'onion',  'onion cooked onion': 'onion',  'onion green onion': 'onion',  'opoponax': 'medicinal',  'orange': 'citrus',  'orange bitter orange': 'bitter',  'orange peel': 'citrus',  'orange rind': 'citrus',  'orangeflower': 'floral',  'orchid': 'floral',  'oriental': 'phenolic',  'origanum': 'orris',  'orris': 'orris',  'osmanthus': 'orris',  'ozone': 'musk',  'painty': 'phenolic',  'palmarosa': 'floral',  'papaya': 'phenolic',  'paper': 'phenolic',  'parsley': 'herbal',  'passion fruit': 'fruity',  'patchouli': 'phenolic',  'pea green pea': 'green',  'peach': 'peach',  'peanut': 'nutty',  'peanut butter': 'buttery',  'peanut roasted peanut': 'nutty',  'pear': 'fruity',  'pear skin': 'fruity',  'peely': 'powdery',  'peony': 'floral',  'pepper bell pepper': 'spicy',  'pepper black pepper': 'spicy',  'peppermint': 'minty',  'peppery': 'spicy',  'petal': 'floral',  'petitgrain': 'pungent',  'petroleum': 'oily',  'phenolic': 'phenolic',  'pine': 'pine',  'pineapple': 'fruity',  'pistachio': 'nutty',  'plastic': 'phenolic',  'plum': 'pine',  'plum skin': 'pine',  'pomegranate': 'fruity',  'popcorn': 'corn',  'pork': 'meaty',  'potato': 'potato',  'potato baked potato': 'potato',  'potato chip': 'potato',  'potato raw potato': 'potato',  'powdery': 'powdery',  'praline': 'chocolate',  'privet': 'coumarinic',  'privetblossom': 'floral',  'prune': 'pine',  'pulpy': 'pungent',  'pumpkin': 'pine',  'pungent': 'pungent',  'quince': 'pungent',  'radish': 'vegetable',  'raisin': 'pine',  'rancid': 'acidic',  'raspberry': 'berry',  'raw': 'rummy',  'reseda': 'anise',  'resinous': 'musty',  'rhubarb': 'rooty',  'rindy': 'rooty',  'ripe': 'fruity',  'roasted': 'roasted',  'root beer': 'rooty',  'rooty': 'rooty',  'rose': 'floral',  'rose dried rose': 'floral',  'rose red rose': 'floral',  'rose tea rose': 'herbal',  'rosemary': 'minty',  'rubbery': 'waxy',  'rue': 'rummy',  'rummy': 'rummy',  'saffron': 'sulfurous',  'sage': 'herbal',  'sage clary sage': 'herbal',  'salty': 'salty',  'sandalwood': 'woody',  'sandy': 'dusty',  'sappy': 'phenolic',  'sarsaparilla': 'jammy',  'sassafrass': 'musty',  'sausage smoked sausage': 'meaty',  'savory': 'salty',  'sawdust': 'dusty',  'scallion': 'onion',  'seafood': 'fishy',  'seashore': 'marine',  'seaweed': 'herbal',  'seedy': 'rooty',  'sharp': 'phenolic',  'shellfish': 'fishy',  'shrimp': 'fishy',  'skunk': 'musk',  'smoky': 'smoky',  'soapy': 'soapy',  'soft': 'sweet',  'solvent': 'solvent',  'soup': 'soapy',  'sour': 'sour',  'spearmint': 'minty',  'spicy': 'spicy',  'spinach': 'vegetable',  'starfruit': 'fruity',  'storax': 'estery',  'strawberry': 'berry',  'styrene': 'sulfurous',  'sugar': 'sweet',  'sugar brown sugar': 'sweet',  'sugar burnt sugar': 'sweet',  'sulfurous': 'sulfurous',  'sweaty': 'musty',  'sweet': 'sweet',  'sweet pea': 'sweet',  'taco': 'tarragon',  'tagette': 'leathery',  'tallow': 'oily',  'tangerine': 'citrus',  'tarragon': 'tarragon',  'tart': 'fruity',  'tea': 'coffee',  'tea black tea': 'coffee',  'tea green tea': 'herbal',  'tequila': 'alcoholic',  'terpenic': 'acetic',  'thujonic': 'thujonic',  'thyme': 'herbal',  'toasted': 'toasted',  'tobacco': 'tobacco',  'toffee': 'chocolate',  'tolu balsam': 'tonka',  'tomato': 'tomato',  'tomato leaf': 'tomato',  'tonka': 'tonka',  'tropical': 'tropical',  'truffle': 'mushroom',  'tuberose': 'floral',  'turkey': 'tarragon',  'turnup': 'phenolic',  'tutti frutti': 'fruity',  'valerian root': 'herbal',  'vanilla': 'vanilla',  'vegetable': 'vegetable',  'verbena': 'herbal',  'vetiver': 'floral',  'vinegar': 'winey',  'violet': 'phenolic',  'violet leaf': 'herbal',  'walnut': 'woody',  'warm': 'cooling',  'wasabi': 'spicy',  'watercress': 'cabbage',  'watermelon': 'melon',  'watermelon rind': 'melon',  'watery': 'musty',  'waxy': 'waxy',  'weedy': 'waxy',  'wet': 'waxy',  'whiskey': 'alcoholic',  'winey': 'winey',  'wintergreen': 'green',  'woody': 'woody',  'woody burnt wood': 'woody',  'woody old wood': 'woody',  'wormwood': 'woody',  'yeasty': 'bready',  'ylang': 'herbal',  'zesty': 'musty'}

def multi_hot(notes,canonical_list):
    notes = [n for n in notes if n in canonical_list]
    indices = torch.tensor([canonical_list.index(n) for n in notes])
    if len(indices) == 0:
        return torch.zeros(len(canonical_list))
        # Occurs when the notes in the pair were removed due to infrequency.
        # raise AttributeError("Found no valid notes.")
    one_hots = torch.nn.functional.one_hot(indices, len(canonical_list))
    return one_hots.sum(dim=0).float()

canonize_using_dict = False

unions = []
intersects = []
model_from_blend_set = []
model_from_single_set = []
ys = []
empty = 0
for d in tqdm.tqdm(full_data):
    blnd = canonize(d["blend_notes"])
    if not blnd:
        empty += 1
        continue
    if canonize_using_dict:
        n1 = set([canon_dict[n] for n in d["mol1_notes"]])
        n2 = set([canon_dict[n] for n in d["mol2_notes"]])
    else:
        n1 = canonize(d["mol1_notes"])
        n2 = canonize(d["mol2_notes"])

    unions.append(multi_hot(n1.union(n2),all_blend_notes))
    intersects.append(multi_hot(n1.intersection(n2),all_blend_notes))
    model_from_blend_set.append(multi_hot(n1,all_blend_notes)+multi_hot(n2,all_blend_notes))
    model_from_single_set.append(multi_hot(n1,all_single_notes)+multi_hot(n2,all_single_notes))
    ys.append(multi_hot(blnd,all_blend_notes))

print(f"Found {empty} empty blends.")

unions = torch.stack(unions)
intersects = torch.stack(intersects)
model_from_blend_set = torch.stack(model_from_blend_set)
model_from_single_set = torch.stack(model_from_single_set)
ys = torch.stack(ys).int()

auroc = torchmetrics.classification.MultilabelAUROC(ys.shape[1])
warnings.filterwarnings("ignore", ".*samples in target*")
print("AUROC for predicting blended labels from unions of constituent labels:",auroc(unions,ys))
print()
print("AUROC for predicting blended labels from intersections of constituent labels:",auroc(intersects,ys))
print()

def train_model(title,model_ds):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(model_ds, ys)
    print(f"Train:{X_train.shape}->{y_train.shape}. Test:{X_test.shape}->{y_test.shape}.")
    lgr = analysis.fingerprint.LogitRegression().fit(X_train,y_train)
    test_pred = torch.from_numpy(lgr.predict(X_test))
    print(title,auroc(test_pred,y_test))
    print()

train_model("AUROC for model from blend -> blend:",model_from_blend_set)
train_model("AUROC for model from single -> blend:",model_from_single_set)


