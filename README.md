# odor-pair
Proof of concept for predicting odor descriptors for molecular pairs.

## Motivation
I am currently looking for a job applying GNNs to research chemistry. My passion is understanding olfaction and my startup [Smellotron](https://smellotron.com/) applied novel NLP techniques to recommend perfumes to inexperienced costumers through free-text queries. This week-long project showcases my ability to adapt previous research techniques to new domains.

As a technical motivation: datasets of well-labeled molecules remain scarce and so novel techniques to more efficiently use the existing datapoints in low data domains lead to much stronger models. It stands as a proof concept for further work in this domain.

## Dataset
Molecular data and odorant label was gathered from an online chemical repository.
Although the dataset contains only ~3k molecules, aromachemical pages contained recommended blenders (molecules that work harmoniously) for specific odors resulting in. Because each molecule's page contained many (~50+) recommendations, over 160k datapoints could be generated.

In many real world applications, the interactions between molecules is often just as important as the properties of the individual molecules themselves. In this dataset, the molecule pair blends often result in new odors that were not apparent in either molecule alone.

![molpair](https://github.com/laurahsisson/odor-pair/assets/10359687/f5a1aec9-4163-4db4-ad20-d62b1189bbc8)

In order to generate a molecule pair datasets, labels for molecule pairs were generated, creating a meta-graph where each node is a molecular graph, and with nodes between edges if the aromachemicals work together well as blenders.

In order to ensure train/test separation, the meta-graph is carved into two components with the following requirements:
Each component must contain datapoints for each label to prevent distributional shift.
Like other carving problems, the edge-boundary degree should be minimized, as data points (pairs) with one molecule in the train dataset and the other molecule in the test dataset must be discarded. As it turns out, minimal width graph carving is NP hard, though [my previous work](https://github.com/laurahsisson/algorithm-ks) shows that special cases can be solved in polynomial time.

Although the dataset contains 109 odor labels, only 33 labels appeared frequently enough (1k pairs) to be included.

Using a randomized carving algorithm, these requirements were met after generating 115,939 training pairs and 3,404 test pairs. Unfortunately, this meant that ~47k datapoints had to be discarded.

## Architecture
A variety of model architectures were tested using a random hyperparameter. The best architecture was structured as follows:
* The Graph Isomorphism message passing step (from [“How Powerful are Graph Neural Networks?”](https://arxiv.org/abs/1810.00826)) was selected for the GNN.
  * The Graph Convolution operator (from [“Semi-supervised Classification with Graph Convolutional Networks”](https://arxiv.org/abs/1609.02907)) and the Kernel-Based Convolutional operator from ([“Neural Message Passing for Quantum Chemistry”](https://arxiv.org/abs/1704.01212)) were also tested, but underperformed compared to the GIN message passing step.
* The GNN was used for three (3) message passing steps, in order to allow weight-tying between these steps, the molecule embeddings were padded to the hidden layer dimension (D=832).
  * The update function used was a simple, 2-layer feedforward neural network.
* The GNN generated embeddings for every atom/node in both molecules across the pair. Graph level readouts were combining the global mean and global add pools across the nodes for each graph.
  * Set2Set readouts were also tested, but no performance improvement was achieved for the significant computational cost increase.
* The graph level embeddings for the two molecules were concattened (in arbitrary order), and then another 2-layer feedwork network generated a pair-level embeddings (also of D=832).
* The final layer predicted logits for all 33 odor labels.

The model was trained for 121 epochs, terminated using early-stopping (patience=0), using an Adam optimizer (lr=2.1x10^-5) with a decaying LR (decay=0.08) across the first 90% of the training. The training routine was adaopted from ([“Neural Message Passing for Quantum Chemistry”](https://arxiv.org/abs/1704.01212)).

## Results
After many hyperparameter trials, the strongest model achieved a mean auroc of 0.800 across all labels. 
The naive 0-R model that uses the frequency of each label across all molecules as the constant prediction (by definition) achieves an auroc of 0.5 for each label. The model performs well for many labels, but barely above random for others. As a comparison, 2048 bit Morgan fingerprints (radius = 4) were generated for each molecule in the pair, then concatenated, and a logistic regression model was fit to predict the odor labels. The model underperforms , i.e. worse than either random or the baseline, on three labels (highlighted in red).
![aurocs](https://github.com/laurahsisson/odor-pair/blob/main/pair_aurocs.png)
The easiest to predict label was "alliaceous" (garlic) reflecting [previous work](https://www.biorxiv.org/content/10.1101/2022.09.01.504602v2) which noted that sulfur containing compounds could be easily be assigned this label. Unlike in this previous work, the model easily predicted ratings for the odor label "musk", with an auroc of 0.925. Because previous work predicted continous ratings for odor, and this dataset contained discrete labels, comparisons are not straightforward. Regardless, the hardest label to predict was "earthy", further research into why different labels are easier or harder to predict based on both the model structure and dataset is necessary. 

## Transfer Learning
Although the model was trained on molecule pairs, it is possible to generate embeddings (dim=832) for individual molecular graphs. In order to predict odor labels for individual molecules, embeddings were generated for the 2362 train and 393 test molecules, and then a logistic regression classifier was trained to predict the same 33 odor labels. The logisitic regression modifier achieved a mean auroc of 0.768, demonstrating that the performance transfers quite well from molecule pair to single molecule prediction. AFinally, a 0-R model with auroc = 0.5 was fit for each label. The same Morgan fingerprints baseline was adapted as before. 
![single_aurocs](https://github.com/laurahsisson/odor-pair/blob/main/single_aurocs.png)
Across every odor label (except one), our model outerperformed standard molecular fingerprints. While the classifier trained on Morgan fingerprints often underperformed compared to random guessing, our model performed better than random across all labels. Though the mean auroc was lower for the transfer learning task, the fact that our model occasionally failed to beat random in the pair prediction task, the fact that our model consistently performed better than random on the single molecule task suggests that the former task is harder than the latter, and that transfer learning is quite effective.

Unsurprisingly, "alliaceous" remained easy to predict for the model. However, "musk" was the easiest label to predict. Based on the structure of the training routine, where all molecules were contained within a meta-graph, "musky" molecules were often conneced and used together as blenders, resulting in similar model embeddings for each molecule in the pair quite naturally, whereas in single molecule training, the model must overcome the structural differences between different categories of "musky" molecules. 

## Room for improvement
* Graph carving is an established space. Using a more sophisticated algorithm, the number of edges could be minimized, increased the dataset size and also allowing more labels to be preserved.

* Hyperparameter search was done randomly, and with limited compute. Using a hyperparameter optimization package like RayTune could provide better results.

* The training task used was a multilabel classification task, but using a triplet loss task would allow more efficient use of the existing datapoints. In the triplet, the anchor would be randomly sampled from all data pairs, the positive would be a data pair with different constituent molecules but a similar odor label, and the negative would be a data pair with similar molecules structurally but a different odor label.

## Future Work
For labels that are difficult to predict because of its commonality across multiple structural classes, researchers and perfumers may benefit from using more specific labels specific to each class. "Musk" is one such label, and to determine if splitting it apart into separate and distinct odor words, researchers could task a panel of experts to determine if two molecules both labelled musk come from the same structural class or distinct ones. If the musks from different classes are easily separable, then new words may be called for.
