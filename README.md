# Olfactory Label Prediction on Aroma-Chemical Pairs
The application of deep learning techniques on aroma-chemicals has resulted in models more accurate than human experts at predicting olfactory qualities. However, public research in this domain has been limited to predicting the qualities of single molecules, whereas in industry applications, perfumers and food scientists are often concerned with blends of many odorants. In this paper, we apply both existing and novel approaches to a dataset we gathered consisting of labeled pairs of molecules. We present graph neural network models capable of generating accurate predictions for the odor qualities arising from blends of aroma-chemicals, and we examine how variations in architecture can lead to significant differences in predictive power.

## Contribution
The contribution of this work is two fold: first, we present a carefully compiled odor mixture dataset wherein the perceptual descriptors of individual components and their blends are available; secondly, we present a set of publicly available GNN based algorithmic frameworks to predict the labels of the mixtures. We also present a number of selected experiments which showcase the efficacy of our approach, and also demonstrate the ability of these models to transfer from single molecule prediction to mixture understanding and vice versa.

## Dataset
Molecular structures (SMILES) and odorant labels were gathered from the Good Scents online chemical repository to generate a dataset of aroma-chemical blends. While the Good Scents website cataloged ~3.5k molecules, each aroma-chemical's page provided suggested blends that, when mixed, yielded distinct aromas. The molecule pages often contained more than 50 blender recommendations, enabling us to gather over 160k molecule-pair data points with discrete labels for 109 olfactory notes.

## Contributors
*Laura Sisson*, ML Engineer @ Talent.com, United States: dataset preparatation, model development

*Aryan Amit Barsainyan*, National Institute of Technology Karnataka, India: ...

*Mrityunjay Sharma*, Academy of Scientific and Innovative Research, India: 

*Ritesh Kumar*, CSIR-CSIO, Chandigarh, India: ...
