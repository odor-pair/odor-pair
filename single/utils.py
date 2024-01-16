import graph.utils

merger_dict = \
{'almond': ['almond','almond bitter almond','almond roasted almond','almond toasted almond'],
 'amber': ['amber', 'ambergris'],
 'apple': ['apple','apple cooked apple','apple dried apple','apple green apple','apple skin'],
 'banana': ['banana','banana peel','banana ripe banana','banana unripe banana'],
 'beefy': ['beefy roasted beefy', 'beef', 'beef juice', 'beefy'],
 'berry': ['berry', 'berry ripe berry'],
 'bitter': ['almond bitter almond', 'bitter', 'orange bitter orange'],
 'black currant': ['currant black currant', 'currant bud black currant bud'],
 'burnt': ['burnt', 'sugar burnt sugar', 'woody burnt wood'],
 'cedar': ['cedar', 'cedarwood'],
 'cheesy': ['cheese','cheesy limburger cheese','cheesy parmesan cheese','cheesy','cheesy roquefort cheese','cheesy feta cheese','cheesy bleu cheese'],
 'cherry': ['cherry', 'cherry maraschino cherry'],
 'chocolate': ['chocolate', 'chocolate dark chocolate'],
 'citrus': ['citralva','citric','citronellal','citrus rind','citronella','citral','citrus','citrus peel'],
 'coffee': ['coffee', 'coffee roasted coffee'],
 'cucumber': ['cucumber', 'cucumber skin'],
 'fresh': ['fresh', 'fresh outdoors', 'freshly'],
 'grape': ['concord grape', 'grape', 'grape skin'],
 'grapefruit': ['grapefruit', 'grapefruit peel', 'grapfruit'],
 'green': ['apple green apple','bean green bean','green','clover','pea green pea','tea green tea'],
 'hay': ['hay', 'hay new mown hay', 'new mown hay'],
 'hazelnut': ['hazelnut', 'hazelnut roasted hazelnut'],
 'honey': ['honey', 'honeydew', 'honeysuckle'],
 'juicy': ['juicy', 'juicy fruit'],
 'lemon': ['lemon peel', 'lime', 'lemon', 'lemongrass'],
 'lily': ['lily of the valley', 'lily-of-the-valley', 'lily', 'lilial'],
 'meaty': ['meaty roasted meaty', 'meat', 'meaty'],
 'melon': ['melon', 'melon rind', 'melon unripe melon', 'watermelon rind'],
 'mint': ['cornmint', 'peppermint', 'mint', 'minty', 'spearmint'],
 'musk': ['musky', 'musk', 'ambrette', 'nitromusk'],
 'odorless': ['almost odorless', 'odorless'],
 'onion': ['onion', 'onion cooked onion', 'onion green onion'],
 'orange': ['orange', 'orange bitter orange', 'orange peel', 'orange rind'],
 'pear': ['pear', 'pear skin'],
 'plum': ['plum', 'plum skin'],
 'potato': ['potato','potato baked potato','potato chip','potato raw potato'],
 'ripe': ['banana ripe banana','banana unripe banana','berry ripe berry','fruit overripe fruit','fruit ripe fruit','ripe'],
 'roasted': ['barley roasted barley','chicken roasted chicken','grain toasted grain','almond roasted almond','peanut roasted peanut','hazelnut roasted hazelnut','coffee roasted coffee','beefy roasted beefy','meaty roasted meaty','roasted'],
 'rose': ['rose red rose','rose','rose tea rose','rose dried rose','bois de rose','rosy','rosey'],
 'tea': ['rose tea rose', 'tea', 'tea black tea', 'tea green tea'],
 'vegetable': ['vegetable', 'vegetables'],
 'woody': ['wood','woody','woody-lactone','woody old wood','woody burnt wood'],
 'grassy': ['lemongrass', 'grassy', 'grass'],
 'lactonic': ['woody-lactone', 'lactonic', 'lactone'],
 'leafy': ['leafy', 'tomato leaf', 'leaf', 'violet leaf'],
 'fruit skin': ['apple skin','grape skin','pear skin','plum skin','orange peel','banana peel','citrus peel','grapefruit peel','lemon peel'],
 'fruity': ['fruit','fruit ripe fruit','juicy fruit','fruit overripe fruit','fruit tropical fruit','fruit dried fruit','fruity'],
 'dry': ['dried','dry','fruit dried fruit','rose dried rose','apple dried apple'],
 'spicy': ['allspice', 'spicy', 'nutmeg', 'spice', 'rosemary'],
 'anisic': ['anise', 'anisic'],
 'balsamic': ['balsam', 'balsamic', 'tolu balsam'],
 'privet': ['privet', 'privetblossom'],
 'camphoreous': ['camphor', 'camphoreous'],
 'caramellic': ['caramel', 'caramellic'],
 'carrot': ['carrot', 'carrot seed'],
 'cooling': ['cool', 'cooling'],
 'coumarinic': ['coumarin', 'coumarinic'],
 'ethereal': ['ether', 'ethereal'],
 'metallic': ['metal', 'metallic'],
 'sulfurous': ['sulfurous', 'sulfury'],
 'terpenic': ['terpene', 'terpenic', 'terpentine'],
 'buttery': ['butter', 'buttery'],
 'cinnamon': ['cinnamon', 'cinnamyl'],
 'fatty': ['chicken fat', 'fatty'],
 'fishy': ['fish', 'fishy', 'shellfish'],
 'herbal': ['herb', 'herbaceous', 'herbal'],
 'milky': ['milk', 'milky'],
 'nutty': ['nut', 'nut flesh', 'nut skin', 'nutty'],
 'oily': ['oil', 'oily'],
 'radish': ['horseradish', 'radish'],
 'rummy': ['rum', 'rummy'],
 'smoky': ['sausage smoked sausage', 'smoky'],
 'bready': ['bread baked', 'bread crust', 'bread rye bread', 'bready'],
 'licorice': ['licorice', 'licorice black licorice'],
 'sweaty': ['sweat', 'sweaty'],
 'winey': ['wine', 'winey'],
 'thujonic': ['absinthe','thujonic'],
 'floral': ['floral', 'flower', 'flowers', 'flowery']}


def count_all_notes_in_merge():
    all_merger_notes = set()
    for n, ns in merger_dict.items():
        all_merger_notes.add(n)
        all_merger_notes.update(ns)

    passed = 0
    failed = 0
    for note in graph.utils.CANON_NOTES_LIST:
        if not note in merger_dict:
            failed += 1
        else:
            passed += 1
    print(passed,failed,passed+failed)

full_canonize = {note: canon for canon, notes in merger_dict.items() for note in notes}
def canonize(notes):
    notes = graph.utils.canonize(notes)
    return sorted(list({full_canonize[note] if note in full_canonize else note for note in notes}))


