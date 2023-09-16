# odor-pair
Proof of concept for predicting odor descriptors for molecular pairs.

## Motivation
I am currently looking for a job applying GNNs to research chemistry. My passion is understanding olfaction and my startup [Smellotron](https://smellotron.com/) applied novel NLP techniques to recommend perfumes to inexperienced costumers through free-text queries. This project stands as a proof concept for further work in this domain, but also showcases my ability to adapt previous research techniques to new domains.

Datasets of well-labeled molecules remain scarce and so novel techniques to more efficiently use the existing datapoints in low data domains lead to much stronger models.



### Dataset
Molecular data and odorant label was gathered from an online chemical repository.
Although the dataset contains only ~2k molecules, aromachemical pages contained recommended blenders (molecules that work harmoniously) for specific odors.
As a result, labels could be constructed for pairs of molecules, generating a dataset of ~150k datapoints.
This creates a meta-graph, where each node is a molecular graph, and nodes are connected if they work well as blenders.

In order to ensure train/test separation, the meta-graph is carved into two components with the following requirements:
Each component must contain datapoints for each label to prevent distributional shift.
Like other carving problems, the edge-boundary degree should be minimized, as data points (pairs) with one molecule in the train dataset and the other molecule in the test dataset must be discarded. As it turns out, minimal width graph carving is NP hard, though [my previous work](https://github.com/laurahsisson/algorithm-ks) shows that special cases can be solved in polynomial time.

Although the dataset contains 109 odor labels, only 33 labels appeared frequently enough (1k pairs) to be included.

Using a randomized carving algorithm, these requirements were met after generating 115939 training data pairs and 3404 test pairs. 

### Results
After many hyperparameter trials, the strongest model achieved an 
![auroc](https://github.com/laurahsisson/odor-pair/assets/10359687/a872a697-7edf-4b7f-a32b-f4018a158212)


### Room for improvement
* Graph carving is an established space. Using a more sophisticated algorithm, the number of edges could be minimized, increased the dataset size and also allowing more labels to be preserved.
* Hyperparameter search was done randomly, and with limited compute. Using a hyperparameter optimization package like RayTune could provide better results.
* The training task used was a multilabel classification task, but using a triplet loss task would allow more efficient use of the existing datapoints. In the triplet, the anchor would be randomly sampled from all data pairs, the positive would be a data pair with different constituent molecules but a similar odor label, and the negative would be a data pair with similar molecules structurally but a different odor label.
