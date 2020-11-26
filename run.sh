#bash run_all_default.sh nn mse microsoft 1.0
#bash run_all_default.sh nn flow_loss2 microsoft 0.1
#bash run_all_default.sh nn flow_loss2 microsoft 1.0
#bash run_all_diff_xgb.sh 6
#bash run_all_diff_xgb.sh 10
#bash run_all_diff_pr.sh mse 0.1 0.00005 10
#bash run_all_diff.sh flow_loss2 1.0 0.00001 15

#ALG=$1
#LOSS_FUNC=$2
#NN_TYPE=$3
#DECAY=$4
#LR=$5
#MAX_EPOCHS=$6

bash run_all_diff.sh flow_loss2 1.0 0.00001 15
bash run_all_diff.sh mse 0.1 0.00005 10

bash run_all_default.sh nn flow_loss2 microsoft 1.0 0.00001 15

#bash run_all_diff_mscn_pr.sh mse 1.0
