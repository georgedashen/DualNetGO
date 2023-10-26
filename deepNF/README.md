# deepNF

deepNF (Gligorijevi´c et al., 2018) is a deep learning model that uses MLP-based autoencoder (MLPAE) for different PPI networks and concatenates latent factors.

More details can be found at the original paper:

[Gligorijević, V., Barot, M., & Bonneau, R. (2018). deepNF: deep network fusion for protein function prediction. Bioinformatics, 34(22), 3873-3881.](https://academic.oup.com/bioinformatics/article/34/22/3873/5026651?ref=https://githubhelp.com)

Original codes can be found at [https://github.com/VGligorijevic/deepNF](https://github.com/VGligorijevic/deepNF).

We provides a pytorch version according to the original codes and description in the paper.

Some data is generated from the main DualNetGO pipeline, so please make sure to run the codes in the **Data preprocessing** section for DualNetGO first.

## Data Preprocessing

First perform random walk with restart (rwr), then generate the ppmi matrix and use minmax normalization for it as the last output.

```
python network_preprocess.py --org human --evidence neighborhood
python network_preprocess.py --org human --evidence fusion
python network_preprocess.py --org human --evidence cooccurence
python network_preprocess.py --org human --evidence coexpression
python network_preprocess.py --org human --evidence experimental
python network_preprocess.py --org human --evidence database
python network_preprocess.py --org human --evidence textmining
```

One can also use the bash script `sh batch_network.sh` to process all adjacency matrices of human and mouse.

## Graph embedding via MDA (MLPAE)

Please make sure that all seven types of PPI networks have been processed before performing the self-supervised learning, as it takes all seven matrices as input and only output one latent vector matrix.

```
python self_supervised_learning_MDA --org human
```

One can change the `--org` argument to _mouse_.

Notice that in the original paper a batch size of 256 is used, but it consumes about 19G memory in GPU. One can use a smaller batch size or change other parameters such as learning_rate if wanted.

## SVM training and evaluation

```
python svmtrain.py
```
