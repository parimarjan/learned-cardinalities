
#LOSS_FUNC=$1
#DECAY=$2
#LR=$3
#MAX_EPOCHS=$4

#HEURISTIC_FEATS=$5
#FLOW_FEATS=$6
#PRED_FEATS=$7
#JOIN_FEATS=$8
#TABLE_FEATS=$9

#bash run_ablation.sh mse 0.1 0.0001 10 0 0 1 1 1
#bash run_ablation.sh mse 0.1 0.0001 10 0 0 1 1 1
#bash run_ablation.sh mse 0.1 0.0001 10 0 0 1 1 1

#bash run_ablation.sh flow_loss2 1.0 0.00001 10 0 0 1 1 1
#bash run_ablation.sh flow_loss2 1.0 0.00001 10 0 0 1 1 1

#bash run_ablation.sh mse 0.1 0.0001 10 0 1 1 1 1
#bash run_ablation.sh flow_loss2 1.0 0.00001 15 0 1 1 1 1

# disable only flow feats
#bash run_ablation.sh mse 0.1 0.0001 10 1 0 1 1 1
#bash run_ablation.sh flow_loss2 1.0 0.00001 15 1 0 1 1 1
#bash run_ablation.sh mse 0.1 0.0001 10 1 0 1 1 1
#bash run_ablation.sh flow_loss2 1.0 0.00001 15 1 0 1 1 1

# disable all except flow feats
#bash run_ablation.sh mse 0.1 0.0001 10 1 1 0 0 0
#bash run_ablation.sh flow_loss2 1.0 0.00001 15 1 1 0 0 0

# disable table+join
#bash run_ablation.sh mse 0.1 0.0001 10 1 1 1 0 0
#bash run_ablation.sh flow_loss2 1.0 0.00001 15 1 1 1 0 0

# no pred feats
#bash run_ablation.sh mse 0.1 0.0001 10 1 1 0 1 1
#bash run_ablation.sh flow_loss2 1.0 0.00001 15 1 1 0 1 1

# no join feats
#bash run_ablation.sh mse 0.1 0.0001 10 1 1 1 0 1
#bash run_ablation.sh flow_loss2 1.0 0.00001 15 1 1 1 0 1

# no table feats
#bash run_ablation.sh mse 0.1 0.0001 10 1 1 1 1 0
bash run_ablation.sh flow_loss2 1.0 0.00001 15 1 1 1 1 0
