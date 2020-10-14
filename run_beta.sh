#BETAS=(0.01 0.1 1.0)
#for i in "${!BETAS[@]}";
  #do
  #python3 main.py --algs nn --nn_type mscn --load_query_together 1 \
  #--normalization_type mscn --max_epochs 30 --tfboard 0 \
  #--save_gradients 1 --loss_func flow_loss2 \
  #--hidden_layer_size 128 --weighted_mse ${BETAS[$i]} \
  #--exp_prefix allTemplates-nli8-wMSE-${BETAS[$i]} --eval_epoch_plan_err 40 \
  #--eval_epoch_flow_err 40 --eval_epoch_jerr 40 --eval_epoch 40 \
  #--result_dir all_results/nli8_weighted_mse/normalized_8b \
  #--normalize_flow_loss 1 \
  #--losses qerr,join-loss,plan-loss,flow-loss \
  #--join_loss_pool_num 40 --cost_model nested_loop_index8b
  #done

HLS=(256 512)
LF=(flow_loss2 mse)

for i in "${!HLS[@]}";
do
  for j in "${!LF[@]}";
  do
  python3 main.py --algs nn --nn_type mscn \
  --normalization_type mscn --max_epochs 30 --tfboard 0 \
  --test_diff_templates 0 \
  --save_gradients 1 --loss_func ${LF[$j]} \
  --hidden_layer_size ${HLS[$i]} \
  --exp_prefix final_results --eval_epoch_plan_err 40 \
  --eval_epoch_flow_err 40 --eval_epoch_jerr 40 --eval_epoch 40 \
  --result_dir all_results/inl_fixed_scan_ops2/nested_loop_index7/final_results \
  --normalize_flow_loss 1 \
  --losses qerr,join-loss,plan-loss,flow-loss \
  --join_loss_pool_num 40 --cost_model nested_loop_index7
  done
done

