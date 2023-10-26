# Graph2GO

Graph2GO (Fan et al., 2020) utilzes variational graph autoencoder (GAE) on the combined version of PPI network and sequence similarity network with protein attributes as input features.

More details can be found at the original paper:

[Fan, K., Guan, Y., & Zhang, Y. (2020). Graph2GO: a multi-modal attributed network embedding method for inferring protein functions. GigaScience, 9(8), giaa081.](https://academic.oup.com/gigascience/article/9/8/giaa081/5885490)

Original codes can be found at [https://github.com/yanzhanglab/Graph2GO](https://github.com/yanzhanglab/Graph2GO).

We provides a pytorch version according to the original codes and description in the paper.

Some data is generated from the main DualNetGO pipeline, so please make sure to run the codes in the **Data preprocessing** section for DualNetGo first.

## Data preprocessing

To implement the VGAE methods, a graph data and a feature matrix is needed as input. As suggested in the paper, one VGAE is used for encoding the graph adjacency matrix, with sequence encoded by the conjoint triad (CT encoding) method, Pfam domain and subcellular location together as the feature matrix; another VGAE is used for encoding the sequence similarity matrix retrieved from blast results, with only Pfam domain and subcellular location together as the feature matrix.

1. For generating the CT encoding:

```
python CTencoder.py --org human --input ../data/9606.protein.sequences.v11.5.fa --output human_CT_343.npy

python CTencoder.py --org mouse --input ../data/10090.protein.sequences.v11.5.fa --output mouse_CT_343.npy
```

2. For generating the sequence similarity matrix:

```
python genFasta.py --input ../data/9606.protein.sequences.v11.5.fa --org human

python genFasta.py --input ../data/10090.protein.sequences.v11.5.fa --org mouse

sh blast.sh

python sim2adj.py --org human --input sim_result_human.txt

python sim2adj.py --org human --input sim_result_human.txt
```

There are around 20000 protein sequences for both human and mouse, so the pairwise alignment across all sequences is time consuming, which takes about 4 hours to complete.

## Graph embedding via VGAE

For reducing the memory consumption on GPU, we use the _NeighborLoader_ provided by _torch_geometric_ to generate subgraphs for batch training, as suggested by the original paper. Here takes the humand dataset as the example.

Encode the PPI adjacency matrix:

```
python GAE_train_loader.py --org human --input ../data/human/human_net_combined.net --type adj
```

Encoder the sequence similarity matrix:

```
python GAE_train_loader.py --org human --input SeqSim_human.mat --type sim
```

## MLP classification and evaluation

A four-layered MLP is used as the prediction head to utilize hidden states from the two matrices. We use the asymmetric loss to deal with sample imbalance across GO terms instead of cross-entropy, which has shown improvement on several multi-label prediction tasks.

```
python classification.py --org human --aspect P --epochs 100
```

For predicting functions from other GO aspects, please change the `--org` and `--aspect` arguments. We reduce the training epoch from 200, which is suggested in the original paper, to 100 to shorten training time. One can change the parameters such as epochs, learning rate and batch size if needed.

One can also use the `output.py` script to output AUPR values for each GO term.
