# None embedding takes about 6G per scripts, others takes 4G

#None
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 100 --step2_iter 30 --max_feat_select 5 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding None &
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 100 --step2_iter 20 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding None &
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 100 --step2_iter 40 --max_feat_select 2 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding None &

#node2vec
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 400 --step2_iter 80 --max_feat_select 3 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding node2vec &
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 200 --step2_iter 20 --max_feat_select 3 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding node2vec &
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 100 --step2_iter 10 --max_feat_select 2 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding node2vec &

#GAE
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 200 --step2_iter 20 --max_feat_select 5 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding GAE &
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 200 --step2_iter 10 --max_feat_select 3 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding GAE &
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 300 --step2_iter 40 --max_feat_select 2 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding GAE &

#MLPAE
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 400 --step2_iter 30 --max_feat_select 3 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding MLPAE &
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 100 --step2_iter 90 --max_feat_select 5 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding MLPAE &
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 400 --step2_iter 80 --max_feat_select 2 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding MLPAE &

#AE
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 100 --step2_iter 90 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect P --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding AE &
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 200 --step2_iter 50 --max_feat_select 3 --num_adj 7 --epochs 100 --aspect F --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding AE &
CUDA_VISIBLE_DEVICES=0 nohup python DualNetGO_mouse.py --step1_iter 400 --step2_iter 10 --max_feat_select 4 --num_adj 7 --epochs 100 --aspect C --dropout1 0.5 --dropout2 0.5 --dropout3 0.1 --lr_fc1 0.01 --lr_fc2 0.01  --hidden 512 --lr_sel 0.01 --embedding AE &

wait
