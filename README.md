# DualNetGO
DualNetGO: A Dual Network Model for Protein Function Prediction via Effective Feature Selection

Here we provide the codes, some of the processed data and important results of the DualNetGO paper. DualNetGO is comprised of two components: a **graph encoder** for extracting graph information or generating graph embeddings and a **predictor** for predicting protein functions.

Most of the codes in this study are bollowed from [CFAGO](http://bliulab.net/CFAGO) and [DualNetGNN](https://github.com/sunilkmaurya/DualNetGNN_large). For more details one can check the original papers at:

[Wu, Z.; Guo, M.; Jin, X.; Chen, J.; Liu, B. CFAGO: Cross-fusion of network and attributes based on attention mechanism for protein function prediction. Bioinformatics 2023, 39, btad123.](https://academic.oup.com/bioinformatics/article/39/3/btad123/7072461)

[Maurya, S.K., Liu, X., Murata, T.: Not all neighbors are friendly: learning to choose hop features to improve node classification. In: International Conference on Information and Knowledgement. (CIKM) (2022)](https://dl.acm.org/doi/abs/10.1145/3511808.3557543)

## Requirements
All experiments are conducted on one 3090 GPU with 24G memory.
```
* python==3.7
* networkx==2.6.3
* numpy==1.21.6
* pandas==1.3.5
* scikit-learn==1.0.2
* scipy==1.7.3
* torch==1.10.1+cu111
* torch-geometric==2.0.0
* torch-sparse==0.6.13
* torch-scatter==2.1.1
* torch-cluster==1.6.1
* torchaudio==0.10.1+cu111
* torchvision==0.11.2+cu111
* termcolor==2.3.0
```

For installing torch-1.10.1+cu111, please go to the pytorch official website and find the corresponding version in the **previous-versions** site. Or you can use the command `pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html` provided by the website.

If installation failed for torch-scatter (also for torch-sparse or torch-cluster), you can try to install with `pip install torch-scatter==2.1.1 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu111.html` and replace the torch and cuda version with your own one.

## Quick run for CAFA3 prediction

**Note: Please make sure pretrained models and processed data are downloaded before performing prediction. They are stored in the cafa3 zip file on zenodo. Put it in the 'data' folder and extract all files.**

We provide the DualNetGO model checkpoint and also the corresponding TransformerAE graph embeddings and Esm sequence embeddings for directly prediction on protein sequences with FASTA file. We use the **blastp** tool from NCBI to search for a most similar sequence in our dataset as a replacement for a sequence that not exists in the any PPI network. So make sure that **blastp** is installed in the environment, or use `sudo apt install ncbi-blast+` to install.

If you have already performed blastp research against the provided dataset and had the `*query_results.txt` file in `--resultdir`, which we have provided, run the following script. You can also run it without the `--txt` argument, then the `--txt` file will be detected automatically in the `--resultdir` folder.

```
CUDA_VISIBLE_DEVICES=0 python DualNetGO_cafa.py --mode predict --aspect C --txt data/cafa3/cc_query_results.txt --resultdir test
```

Or use the following script if you want to use a custom `--fasta` file as input, with an example we provided:

```
CUDA_VISIBLE_DEVICES=0 python DualNetGO_cafa.py --mode predict --aspect C --fasta data/cafa3/cc-test.fasta --resultdir test
```

All feature matrices for sequences in the `--fasta` file will be gathered according to the blastp results and stored in the `--resultdir` folder. Two result files are generated: one for the **score matrix** in `.npy` format and one for **tabular output** containing query sequence ids in fasta file, GO terms and prediction scores as columns. The second file is ready for CAFA competition submission.

## Reproducing results for human/mouse with single-species models

**Note: Before running following codes, please make sure corresponding [processed_data](https://zenodo.org/records/10526397) has been downloaded, extracted, and placed in the 'data' folder.**

For reproducing the results of human reported in the paper using graph embeddings from TransformerAE:

```
# using GPU
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 300 --step2_iter 10 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding AE --modeldir human_best --out results_human_best.csv

CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 100 --step2_iter 40 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding AE --modeldir human_best --out results_human_best.csv

CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 500 --step2_iter 80 --max_feat_select 3 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01 --hidden 512 --lr_sel 0.01 --embedding AE --modeldir human_best --out results_human_best.csv

# Fmax Results: BP 0.459, MF 0.226, CC 0.464
# Best masks: BP [3,5,6], MF [3,4,6], CC [0,2]
```

For reproducing the results of mouse reported in the paper:

```
CUDA_VISIBLE_DEVICES=0 python DualNetGO_mouse.py --step1_iter 100 --step2_iter 90 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding AE --modeldir mouse_best --out results_mouse_best.csv

CUDA_VISIBLE_DEVICES=0 python DualNetGO_mouse.py --step1_iter 200 --step2_iter 50 --max_feat_select 3 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding AE --modeldir mouse_best --out results_mouse_best.csv

CUDA_VISIBLE_DEVICES=0 python DualNetGO_mouse.py --step1_iter 400 --step2_iter 10 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01 --hidden 512 --lr_sel 0.01 --embedding AE --modeldir mouse_best --out results_mouse_best.csv

# Fmax Results: BP 0.296, MF 0.524, CC 0.502
# Best masks: BP [0,3,6], MF [0,4], CC [1,2,6]
```

After running the above, three models for BP, MF and CC, respectively, are trained, and their weights and the corresponding results containing the best masks are saved. 


## 1. Data preprocessing
Raw PPI network data can be downloaded from the STRING database and protein attribute information can be retrieved from the Swiss-Prot database. If one has already downloaded the processed data (with raw data included) from above, then may skip this step.

One can also get the files used in the paper from the [_CFAGO_](http://bliulab.net/CFAGO/static/dataset/Dataset.rar) website and save them in the `data` folder, but please notice that they are not the latest version now. 

**Important note**: 

**There are substantial updates of STRING v12.0 from v11.5, including how STRING ID is named. Latest files downloaded for Pfam and subcellular location annotation from Uniprot have been using the STRING ID from STRING v12.0, and column names are slightly different from those used in this paper, which are original provided by the CFAGO paper.**

For generating adjacency matrices from PPI networks, retrieving protein Pfam domain and subcellular location and filtering GO terms and proteins. Take human data as the example:

```
cd prepocessing

python annotation_preprocess.py -data_path ../data -af goa_human.gaf -pf 9606.protein.info.v11.5.txt -ppif 9606.protein.links.detailed.v11.5.txt.gz -org human -stl 41

python network_data_preprocess.py -data_path ../data -snf 9606.protein.links.detailed.v11.5.txt.gz -org human -n combined
python network_data_preprocess.py -data_path ../data -snf 9606.protein.links.detailed.v11.5.txt.gz -org human -n neighborhood
python network_data_preprocess.py -data_path ../data -snf 9606.protein.links.detailed.v11.5.txt.gz -org human -n fusion
python network_data_preprocess.py -data_path ../data -snf 9606.protein.links.detailed.v11.5.txt.gz -org human -n cooccurence
python network_data_preprocess.py -data_path ../data -snf 9606.protein.links.detailed.v11.5.txt.gz -org human -n coexpression
python network_data_preprocess.py -data_path ../data -snf 9606.protein.links.detailed.v11.5.txt.gz -org human -n experimental
python network_data_preprocess.py -data_path ../data -snf 9606.protein.links.detailed.v11.5.txt.gz -org human -n database
python network_data_preprocess.py -data_path ../data -snf 9606.protein.links.detailed.v11.5.txt.gz -org human -n textmining

python attribute_data_preprocess.py -data_path ../data -pf 9606.protein.info.v11.5.txt -ppif 9606.protein.links.detailed.v11.5.txt.gz -org human -uniprot 'uniprot-filtered-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[96--.tab'
```

The resulting files are `human_annot.mat` for datasets and labels, `human_net_*.mat` for adjacency matices, and `features.npy` for protein attributes.

For mouse please use the data with **10090** taxonomy code, and `mgi.gaf` as the annotation file. Uniprot file for mouse is `uniprot-download_true_fields_accession_2Creviewed_2Csequence_2Cxref_-2022.06.29-08.34.18.65`.

## 2. Graph embedding

While there are many options for getting useful information from graphs, in this study we use a transformer-based autoencoder (**TransformerAE**) introduced by the _CFAGO_ paper. The TransformerAE takes the raw adjacency matrix of PPI network (minmax-normalized weighted vectors) and the protein attribute matrix (one-hot vectors of domain and subcellular location) as input, passes them through 6 attention encoder layers, gets a low-dimension hidden state matrix, and then passes it through another 6 attention encoder (without masks) layers to reconstruct the original adjacency matrix and attribute matrix. The hidden state matrix is used for graph embeddings for the PPI network with respect to a specific type of evidence. 

Most of the codes are the same as provided in the CFAGO repository except a little modification for attaining the hidden state matrix after tranining. Noted that we use completely the default setting provided in the original code and find out that it very time consuming, about **50 hours** as using 5000 epochs! We have tried a reduced epoch number such as 500 and performance on prediction is even better for CFAGO (not reported in the script), so it is fine to use a smaller epoch number to shorten training time.

```
cd CFAGO

python self_supervised_leaning.py --org human --dataset_dir ../data/human --output human_result --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5 --evidence combined
python self_supervised_leaning.py --org human --dataset_dir ../data/human --output human_result --dist-url tcp://127.0.0.1:3724 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5 --evidence neighborhood
python self_supervised_leaning.py --org human --dataset_dir ../data/human --output human_result --dist-url tcp://127.0.0.1:3725 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5 --evidence fusion
python self_supervised_leaning.py --org human --dataset_dir ../data/human --output human_result --dist-url tcp://127.0.0.1:3726 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5 --evidence cooccurence
python self_supervised_leaning.py --org human --dataset_dir ../data/human --output human_result --dist-url tcp://127.0.0.1:3727 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5 --evidence coexpression
python self_supervised_leaning.py --org human --dataset_dir ../data/human --output human_result --dist-url tcp://127.0.0.1:3728 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5 --evidence experimental
python self_supervised_leaning.py --org human --dataset_dir ../data/human --output human_result --dist-url tcp://127.0.0.1:3729 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5 --evidence database
python self_supervised_leaning.py --org human --dataset_dir ../data/human --output human_result --dist-url tcp://127.0.0.1:3730 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5 --evidence textmining
```

There is a design in the original code for parallelly training across GPUs and machines but we doesn't use it in the traninig. When running multiple processes, please make sure using different `--dist-url` such as _tcp://127.0.0.1:3724_. The `--evidence` argument correspond to the type of PPI network to encode, and can be chosen from `neighborhood`, `fusion`, `cooccurence`, `coexpression`, `experimental`, `database`, `textmining` and `combined`.

Those who are interested in classic graph embedding methods can see below:

```
#for node2vec and graph variational autoencoder
cd preprocessing

python node2vec_train.py --org human --evidence combined

python GAE_train_sampler.py --org human --evidence combined
```

```
#for MLPAE
cd CFAGO

python self_supervised_leaning_MLPAE.py --org human --dataset_dir ../data/human --output human_MLPAE_result --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 100 --lr 1e-5 --evidence combined
```

## 3. Training and Prediction

For training DualNetGO model:

```
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --org human --step1_iter 100 --step2_iter 50 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01 --hidden 512 --lr_sel 0.01

CUDA_VISIBLE_DEVICES=0 python DualNetGO_mouse.py --org mouse --step1_iter 100 --step2_iter 50 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01 --hidden 512 --lr_sel 0.01
```

`CUDA_VISIBLE_DEVICES=0` specifies the GPU card to use. `step1_iter` and `step2_iter` are the epoch number for stage 1 and stage 2, respectively. `epochs` controls the epoch number for stage 3, which is the summed number of epochs for stage 2 and 3.

For testing the evidence-centric model, use `DualNetGO_evidence.py` and make sure that all four embeddings include AE, MLPAE, node2vec and GAE exist. 

```
CUDA_VISIBLE_DEVICES=0 python DualNetGO_evidence.py --org human --aspect P --evidence combined --step1_iter 100 --step2_iter 50 --epochs 100 --max_feat_select 4 --num_adj 5 --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01 --hidden 512 --lr_sel 0.01
```

The `DualNetGO_output.py` is used for generating additional AUPR values of each GO terms.

For those who are interested in reproducing the best results with TransformerAE or other embedding methods, run `sh experiment_best.sh` for human and `sh experiment_best_mouse.sh` for mouse. Remember to modify the proper device id in `CUDA_VISIBLE_DEVICES=[device_id]` to specify which gpu card to be used.

## 4. Other network-based models

As some of the famous network-based methods were derived early and not implemented by pytorch framework, we also provide modified pytorch version for these methods, which include [_deepNF_](https://github.com/VGligorijevic/deepNF) and [_Graph2GO_](https://github.com/yanzhanglab/Graph2GO). The [_Mashup_](http://mashup.csail.mit.edu) method was implmented via MATLAB, but we are not able to convert it into a pytorch version. We modify the MATLAB code of Mashup to only attain the diffusion hidden states and train it using python with SVM kernel method. Example codes for two baseline models _Naive_ and _Blast_ are also provided.
