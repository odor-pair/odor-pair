# odor-pair
Proof of concept for predicting odor descriptors for molecular pairs.

## Motivation
I am currently looking for a job applying GNNs to research chemistry. My passion is understanding olfaction and my startup [Smellotron](https://smellotron.com/) applied novel NLP techniques to recommend perfumes to inexperienced costumers through free-text queries. This week-long project showcases my ability to adapt previous research techniques to new domains.

As a technical motivation: datasets of well-labeled molecules remain scarce and so novel techniques to more efficiently use the existing datapoints in low data domains lead to much stronger models. It stands as a proof concept for further work in this domain.



## Dataset
Molecular data and odorant label was gathered from an online chemical repository.
Although the dataset contains only ~20k molecules, aromachemical pages contained recommended blenders (molecules that work harmoniously) for specific odors resulting in. Because each molecule's page contained many (~5+) recommendations, over 160k datapoints could be generated.

In many real world applications, the interactions between molecules is often just as important as the properties of the individual molecules themselves. In this dataset, the molecule pair blends often result in new odors that were not apparent in either molecule alone.

![molpair](https://github.com/laurahsisson/odor-pair/assets/10359687/f5a1aec9-4163-4db4-ad20-d62b1189bbc8)

In order to generate a molecule pair datasets, labels for molecule pairs were generated, creating graph where each node is a molecular, and nodes are connected if they work well as blenders.

In order to ensure train/test separation, the meta-graph is carved into two components with the following requirements:
Each component must contain datapoints for each label to prevent distributional shift.
Like other carving problems, the edge-boundary degree should be minimized, as data points (pairs) with one molecule in the train dataset and the other molecule in the test dataset must be discarded. As it turns out, minimal width graph carving is NP hard, though [my previous work](https://github.com/laurahsisson/algorithm-ks) shows that special cases can be solved in polynomial time.

Although the dataset contains 109 odor labels, only 33 labels appeared frequently enough (1k pairs) to be included.

Using a randomized carving algorithm, these requirements were met after generating 115,939 training pairs and 3,404 test pairs. Unfortunately, this meant that ~47k datapoints had to be discarded.

## Results
After many hyperparameter trials, the strongest model achieved a mean auroc of 0.800 across all labels. 
The naive 0-R model that uses the frequency of each label across all molecules as the constant prediction (by definition) achieves an auroc of 0.5 for each label. The model performs well for many labels, but barely above random for others. The model significantly underperforms for one label.
![aurocs](https://github.com/laurahsisson/odor-pair/assets/10359687/c17615c0-57a8-43bb-8ca6-b6c78b221870)
The easiest to predict label was "alliaceous" (garlic) reflecting [previous work](https://www.biorxiv.org/content/10.1101/2022.09.01.504602v2) which noted that sulfur containing compounds could be easily be assigned this label. Unlike in this previous work, our model easily predicted ratings for the odor label "musk", with an auroc of 0.925. The hardest label to predict was "earthy", further research into why different labels are easier or harder to predict based on both the model structure and dataset is necessary.

## Room for improvement
* Graph carving is an established space. Using a more sophisticated algorithm, the number of edges could be minimized, increased the dataset size and also allowing more labels to be preserved.

* Hyperparameter search was done randomly, and with limited compute. Using a hyperparameter optimization package like RayTune could provide better results.

* The training task used was a multilabel classification task, but using a triplet loss task would allow more efficient use of the existing datapoints. In the triplet, the anchor would be randomly sampled from all data pairs, the positive would be a data pair with different constituent molecules but a similar odor label, and the negative would be a data pair with similar molecules structurally but a different odor label.

## Future Work
For labels that are difficult to predict because of its commonality across multiple structural classes, researchers and perfumers may benefit from using more specific labels specific to each class. "Musk" is one such label, and to determine if splitting it apart into separate and distinct odor words, researchers could task a panel of experts to determine if two molecules both labelled musk come from the same structural class or distinct ones. If the musks from different classes are easily separable, then new words may be called for.
