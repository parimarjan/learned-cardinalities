
#LOSS_FUNC=$1
#DECAY=$2
#FLOW_FEATS=$3

#TEST_DIFF=$4
#EVAL_ON_JOB=$5
#HEURISTIC_FEATS=$6

#REL_ESTS=$7
#ONEHOT=$8

#JOIN_FEATS=1
#TABLE_FEATS=1
#PRED_FEATS=1

# default, evaluating on JOB as well w/o heuristic feats etc.
#bash run_all_ablation_fcnn.sh flow_loss2 1.0 1 0 1 0 0 0
#bash run_all_ablation_fcnn.sh flow_loss2 1.0 0 0 1 0 0 0

# flow feats, but no heuristics / pg ests etc.
bash run_all_ablation_fcnn.sh flow_loss2 1.0 1 1 0 0 0 0

# No flow feats
bash run_all_ablation_fcnn.sh flow_loss2 1.0 0 1 0 1 0 0

# No one-hot, or rel feats
bash run_all_ablation_fcnn.sh flow_loss2 1.0 1 1 0 1 0 0

# No one-hot feats
bash run_all_ablation_fcnn.sh flow_loss2 1.0 1 1 0 1 1 0
