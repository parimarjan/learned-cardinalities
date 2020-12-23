
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



### wander join stuff
#bash run_wanderjoin.sh flow_loss2 1.0 0.00001 15 wanderjoin
#bash run_wanderjoin.sh mse 0.1 0.0001 10 wanderjoin

#bash run_wanderjoin.sh flow_loss2 1.0 0.00001 15 actual

#LOSS_FUNC=$1
#DECAY=$2
#LR=$3
#MAX_EPOCHS=$4
#bash run_all_diff.sh flow_loss2 1.0 0.00001 15
#bash run_all_diff.sh mse 0.1 0.0001 15

#time python3 scripts/get_query_cardinalities.py --query_dir \
#our_dataset/queries/9b -n -1 --num_proc 4 --card_type wanderjoin \
#--wj_walk_timeout 2.0

#time python3 scripts/get_query_cardinalities.py --query_dir \
#our_dataset/queries/11a -n -1 --num_proc 4 --card_type wanderjoin \
#--wj_walk_timeout 4.0

#time python3 scripts/get_query_cardinalities.py --query_dir \
#our_dataset/queries/11b -n -1 --num_proc 4 --card_type wanderjoin \
#--wj_walk_timeout 2.0
