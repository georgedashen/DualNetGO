# CFAGO

CFAGO (Wu et al., 2023) designs a transformer-based autoencoder (TransformerAE) to cross-fuse the combined PPI graph and protein attributes.

More details can be found at the original paper: 

[Wu, Z.; Guo, M.; Jin, X.; Chen, J.; Liu, B. CFAGO: Cross-fusion of network and attributes based on attention mechanism for protein function prediction. Bioinformatics 2023, 39, btad123.](https://academic.oup.com/bioinformatics/article/39/3/btad123/7072461)

Original codes can data can be downloaded from: [CFAGO data and codes]((http://bliulab.net/CFAGO)).

We modify the code a little bit to restrict training only on the training set. Validation set is used for evaluating when the model gets the best Fmax score and determining a threshold to decide whether a function. Evaluation on test set with parameters when the model attains best Fmax on validation set is reported as the final results.

## Preprocessing

We add an argument `--evidence` to allow generating adjacency matrix of other evidence such as _coexpression_ and _experimental_.

```
python annotation_preprocess.py -data_path ../data -af goa_human.gaf -pf 9606.protein.info.v11.5.txt -ppif 9606.protein.links.detailed.v11.5.txt -org human -stl 41

python network_data_preprocess.py -data_path ../data -snf 9606.protein.links.detailed.v11.5.txt -org human --evidence combined

python attribute_data_preprocess.py -data_path ../data -pf 9606.protein.info.v11.5.txt -ppif 9606.protein.links.detailed.v11.5.txt -org human -uniprot uniprot-filtered-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[96--.tab
```

## Graph embedding via multi-head self-attention layers (TransformerAE)

We add an argument `--evidence` to allow generating graph embeddings for other evidence.

```
python self_supervised_leaning.py --org human --dataset_dir ../data/human --output human_result --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5 --evidence combined
```

## Fine-tuning for protein function prediction

We also add an argument `--evidence` to specify which pretraining model to use with respect to PPI evidence.

```
python CFAGO_split_comet.py --org human --dataset_dir ../data/human --output human_result --evidence combined --aspect P --num_class 45 --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.3 --attention_layers 6 --gamma_pos 0 --gamma_neg 2 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-4 --pretrained_model human_result/human_attention_layers_6_lr_1e-05_seed_1329765522_activation_gelu_model_combined.pkl
```
