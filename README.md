# odor-pair
Proof of concept for predicting odor descriptors for molecular pairs.


## Dataset
Molecular data and odorant label was gathered from an online chemical repository.
Although the dataset contains only ~2k molecules, aromachemical pages contained recommended blenders (molecules that work harmoniously) for specific odors.
As a result, labels could be constructed for pairs of molecules, generating a dataset of ~150k datapoints.
This creates a meta-graph, where each node is a molecular graph, and nodes are connected if they work well as blenders.

In order to ensure train/test separation, the meta-graph is carved into two components with the following requirements:
Each component must contain datapoints for each label to prevent distributional shift.
Like other carving problems, the edge-boundary degree should be minimized, as data points (pairs) with one molecule in the train dataset and the other molecule in the test dataset must be discarded.

Using a randomized carving algorithm, these requirements were met after generating 115939 training data pairs and 3404 test pairs. 
