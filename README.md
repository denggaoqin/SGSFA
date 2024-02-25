# SGSFA: A Subgraph Representation Learning by Self-attention Free Adversarial Training

#### Requirements
Tested combination: Python 3.9.6 + [PyTorch 11.6](https://pytorch.org/get-started/previous-versions/) + [PyTorch_Geometric 1.7.2](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Other required python libraries include: numpy, scikit-learn, pyyaml etc.


#### Prepare Data

You can download the realworld datasets [here](https://www.dropbox.com/sh/zv7gw2bqzqev9yn/AACR9iR4Ok7f9x1fIAiVCdj3a?dl=0) or from the [mirror](https://disk.pku.edu.cn/#/link/B85C0589ADE44E0CFF8AAD6A4D6BF6B0%20). Please download, unzip, and put them in ./dataset/.

The location of each dataset should be
```
CODE
├── dataset
│   ├── em_user
│   ├── hpo_metab
│   ├── ppi_bp
│   └── hpo_neuro
└── dataset_
    └── your_dataset

```
#### Reproduce SGSFA

To reproduce our results on real-world datasets:

We have provided our SSL embeddings in ./Emb/. You can also reproduce them by
```
python GNNEmb.py --use_nodeid --device $gpu_id --dataset $dataset --name $dataset
```
Then 
```
python SGSFATest.py --use_nodeid --use_seed --use_maxzeroone --repeat 10 --device $gpu_id --dataset $dataset
```
where $dataset can be selected from em_user, ppi_bp, hpo_metab, and hpo_neuro.

To reproduce GNN-seg
```
python GNNSeg.py --test  --repeat 10 --device $gpu_id --dataset $dataset
```
#### Use Your Own Dataset

Please add a branch in the `load_dataset` function in datasets.py to load your dataset and create a configuration file in ./config to describe the hyperparameters for the SGSFA model.
