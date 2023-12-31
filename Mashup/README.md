# Mashup

Mashup (Cho et al., 2016) generates balanced hidden diffusion states across different PPI networks.

More details can be found at the original paper:

[Cho, H., Berger, B., & Peng, J. (2016). Compact integration of multi-network topology for functional analysis of genes. Cell systems, 3(6), 540-548.](https://www.sciencedirect.com/science/article/pii/S240547121630360X?via%3Dihub)

Original codes can be downloaded from: [Mashup data and codes](https://cb.csail.mit.edu/cb/mashup/).

We modify the code a little bit to only use Mashup to generate the diffusion hidden states and then use python to train a kernel svm on these hidden states.

Some data is generated from the main DualNetGO pipeline, so please make sure to run the codes in the **Data preprocessing** section for DualNetGO first.

## Preprocess adjacency matrix

First use python to process the PPI adjacency matrix into adjacency edge list. PPI from all seven evidence must all be processed before next step.

```
python genEdgeList.py --org human --evidence neighborhood &
python genEdgeList.py --org human --evidence fusion &
python genEdgeList.py --org human --evidence cooccurence &
python genEdgeList.py --org human --evidence coexpression &
python genEdgeList.py --org human --evidence experimental &
python genEdgeList.py --org human --evidence database &
python genEdgeList.py --org human --evidence textmining &
```

One may change the `--org` argument from _human_ to _mouse_ to generate data for mouse.

## Generate diffusion hidden states

Please make sure you have a MATLAB installed. For using GUI version just open MATLAB directly and run the `example_script.m`, or one can also run it through a terminal.

## SVM training and evaluation

To predict GO terms from the BP functional category of human, run:

```
python trainsvm.py --org human --aspect P
```
