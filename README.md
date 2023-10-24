# DualNetGO
DualNetGO: A Dual-network Deep Learning Model to Intelligently Select Multi-modal Features for Protein Function Prediction

Here we provide the codes, some of the processed data and important results of the DualNetGO paper. DualNetGO is comprised of two components: a **graph encoder** for extracting graph information or generating graph embeddings and a **predictor** for predicting protein functions.

Most of the codes in this study are bollowed from [CFAGO](http://bliulab.net/CFAGO) and [DualNetGNN](https://github.com/sunilkmaurya/DualNetGNN_large). For more details one can check the original papers at:

Wu, Z.; Guo, M.; Jin, X.; Chen, J.; Liu, B. CFAGO: Cross-fusion of network and attributes based on attention mechanism for protein function prediction. Bioinformatics 2023, 39, btad123.(https://academic.oup.com/bioinformatics/article/39/3/btad123/7072461)

Maurya, S.K., Liu, X., Murata, T.: Not all neighbors are friendly: learning to choose hop features to improve node classification. In: International Conference on Information and Knowledgement. (CIKM) (2022) [https://dl.acm.org/doi/abs/10.1145/3511808.3557543]

## Requirements
All experiments are conducted on one 3090 GPU with 24G memory.
```
* keras==2.11.0
* numpy==1.21.6
* pandas==1.3.5
* scikit-learn==1.0.2
* scipy==1.7.3
* Theano==1.0.5
* torch==1.10.1+cu111
* tqdm==4.64.1
```
## 1. Data Preprocessing

Raw PPI network data can be downloaded from the STRING database and protein attribute information can be retrieved from the Swiss-Prot database. One can also get the files used in the paper from the google drive and save them in the `Data` folder, but please notice that they are not the latest version now. For generating adjacency matrices from PPI networks, retrieving protein Pfam domain and subcellular location and filtering GO terms and proteins, take human data as the example:

```
cd prepocessing
python annotation_preprocess.py -data_path ../Dataset -af goa_human.gaf -pf 9606.protein.info.v11.5.txt -ppif 9606.protein.links.detailed.v11.5.txt -org human -stl 41
python network_data_preprocess.py -data_path ../Dataset -snf 9606.protein.links.detailed.v11.5.txt -org human
python attribute_data_preprocess.py -data_path ../Dataset -pf 9606.protein.info.v11.5.txt -ppif 9606.protein.links.detailed.v11.5.txt -org human -uniprot uniprot-filtered-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[96--.tab
```

## 2. Graph Embedding

While there are many options for getting useful information from graphs, in this study we use a transformer-based autoencoder (**TransformerAE**) introduced by the _CFAGO_ paper. The TransformerAE takes the raw adjacency matrix of PPI network (minmax-normalized weighted vectors) and the protein attribute matrix (one-hot vectors of domain and subcellular location) as input, passes them through 6 attention encoder layers, gets a low-dimension hidden state matrix, and then passes it through another 6 attention encoder (without masks) layers to reconstruct the original adjacency matrix and attribute matrix. The hidden state matrix is used for graph embeddings for the PPI network with respect to a specific type of evidence. 

Most of the codes are the same as provided in the CFAGO repository except a little modification for attaining the hidden state matrix after tranining. Noted that we use completely the default setting provided in the original code and find out that it very time consuming (about 50 hours as using 5000 epochs). We have tried a reduced epoch number such as 500 and performance on prediction is even better for CFAGO, so it is fine to use a smaller epoch number to shorten training time.

```
cd CFAGO
python self_supervised_leaning.py --org human --dataset_dir ../Dataset/human --output human_result --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5 --evidence combined
```

There is a design in the original code for parallelly training across GPUs and machines but we doesn't use it in the traninig. When running multiple processes, please make sure using different `--dist-url` such as tcp://127.0.0.1:3724. The `--evidence` argument correspond to the type of PPI network to encode, and can be chosen from `neighborhood`, `fusion`, `cooccurence`, `coexpression`, `experimental`, `database`, `textmining` and `combined`.

