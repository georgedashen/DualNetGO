# None embedding takes about 6G per scripts, others takes 4G

#None
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 400 --step2_iter 70 --max_feat_select 3 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding None --modeldir human_best --out results_human_best.csv &
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 100 --step2_iter 30 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding None --modeldir human_best --out results_human_best.csv &
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 300 --step2_iter 30 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding None --modeldir human_best --out results_human_best.csv &

#node2vec
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 100 --step2_iter 50 --max_feat_select 2 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding node2vec --modeldir human_best --out results_human_best.csv &
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 100 --step2_iter 50 --max_feat_select 3 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding node2vec --modeldir human_best --out results_human_best.csv &
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 500 --step2_iter 70 --max_feat_select 3 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding node2vec --modeldir human_best --out results_human_best.csv &

#GAE
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 200 --step2_iter 50 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding GAE --modeldir human_best --out results_human_best.csv &
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 400 --step2_iter 50 --max_feat_select 2 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding GAE --modeldir human_best --out results_human_best.csv &
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 400 --step2_iter 90 --max_feat_select 5 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding GAE --modeldir human_best --out results_human_best.csv &

#MLPAE
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 300 --step2_iter 70 --max_feat_select 2 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding MLPAE --modeldir human_best --out results_human_best.csv &
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 500 --step2_iter 40 --max_feat_select 2 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding MLPAE --modeldir human_best --out results_human_best.csv &
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 500 --step2_iter 20 --max_feat_select 2 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding MLPAE --modeldir human_best --out results_human_best.csv &

#AE
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 300 --step2_iter 10 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding AE --modeldir human_best --out results_human_best.csv &
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 100 --step2_iter 40 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding AE --modeldir human_best --out results_human_best.csv &
CUDA_VISIBLE_DEVICES=0 python DualNetGO.py --step1_iter 500 --step2_iter 80 --max_feat_select 3 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding AE --modeldir human_best --out results_human_best.csv &

wait
