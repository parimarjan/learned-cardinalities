#bash run_all_default.sh nn mse microsoft 1.0
#bash run_all_default.sh nn flow_loss2 microsoft 0.1
#bash run_all_default.sh nn flow_loss2 microsoft 1.0

#bash run_all_default.sh nn mse mscn_set 0.1
#bash run_all_default.sh nn mse mscn_set 1.0
#bash run_all_diff_mscn.sh nn mse mscn_set
bash run_all_diff.sh nn mse mscn_set
