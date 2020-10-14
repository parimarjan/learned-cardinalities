MAX_EPOCHS=20

ALG=nn
LOSS_FUNC=flow_loss2
NN_TYPE=mscn

#BUCKETS=(10)
#HLS=(64 128 256)
BUCKETS=10
HLS=(512)
#MIN_QERRS=(2.0 4.0 8.0 16.0 32.0 64.0)
NUM_PAR=40

EVAL_EPOCH=40
SAMPLE_BITMAP=0

#CMD="time python3 main.py --algs nn -n -1 \
 #--max_discrete_featurizing_buckets 1 \
 #--hidden_layer_size 512 \
 #--weight_decay 0.1 \
 #--alg $ALG \
 #--loss_func $LOSS_FUNC \
 #--nn_type $NN_TYPE \
 #--sample_bitmap $SAMPLE_BITMAP \
 #--test_size 0.5 \
 #--exp_prefix final_runs \
 #--result_dir \
  #all_results/inl_fixed_scan_ops/nested_loop_index7/default/ablation \
 #--max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
 #--eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
 #--optimizer_name adamw \
 #--normalize_flow_loss 1 \
 #--eval_on_job 1 \
 #--feat_rel_pg_ests  1 \
 #--feat_rel_pg_ests_onehot  1 \
 #--feat_pg_est_one_hot  1 \
 #--flow_features 1 --feat_tolerance 0 \
 #--lr 0.0001"
#echo $CMD
#eval $CMD

#CMD="time python3 main.py --algs nn -n -1 \
 #--max_discrete_featurizing_buckets 10 \
 #--hidden_layer_size 512 \
 #--weight_decay 0.1 \
 #--alg $ALG \
 #--loss_func $LOSS_FUNC \
 #--nn_type $NN_TYPE \
 #--sample_bitmap $SAMPLE_BITMAP \
 #--test_size 0.5 \
 #--exp_prefix final_runs \
 #--result_dir \
  #all_results/inl_fixed_scan_ops/nested_loop_index7/default/ablation \
 #--max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
 #--eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
 #--optimizer_name adamw \
 #--normalize_flow_loss 1 \
 #--eval_on_job 1 \
 #--feat_rel_pg_ests  1 \
 #--feat_rel_pg_ests_onehot  1 \
 #--feat_pg_est_one_hot  1 \
 #--flow_features 0 --feat_tolerance 0 \
 #--lr 0.0001"
#echo $CMD
#eval $CMD

CMD="time python3 main.py --algs nn -n -1 \
 --max_discrete_featurizing_buckets 10 \
 --hidden_layer_size 512 \
 --weight_decay 0.1 \
 --alg $ALG \
 --loss_func $LOSS_FUNC \
 --nn_type $NN_TYPE \
 --sample_bitmap $SAMPLE_BITMAP \
 --test_size 0.5 \
 --exp_prefix final_runs \
 --result_dir \
  all_results/inl_fixed_scan_ops/nested_loop_index7/default/ablation \
 --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
 --eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
 --optimizer_name adamw \
 --normalize_flow_loss 1 \
 --eval_on_job 1 \
 --heuristic_features 0 \
 --feat_rel_pg_ests  0 \
 --feat_rel_pg_ests_onehot  0 \
 --feat_pg_est_one_hot  0 \
 --flow_features 1 --feat_tolerance 0 \
 --lr 0.0001"
echo $CMD
eval $CMD

CMD="time python3 main.py --algs nn -n -1 \
 --max_discrete_featurizing_buckets 10 \
 --hidden_layer_size 512 \
 --weight_decay 0.1 \
 --alg $ALG \
 --loss_func $LOSS_FUNC \
 --nn_type $NN_TYPE \
 --sample_bitmap $SAMPLE_BITMAP \
 --test_size 0.5 \
 --exp_prefix final_runs \
 --result_dir \
  all_results/inl_fixed_scan_ops/nested_loop_index7/default/ablation \
 --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
 --eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
 --optimizer_name adamw \
 --normalize_flow_loss 1 \
 --eval_on_job 1 \
 --join_features 0 \
 --heuristic_features 1 \
 --feat_rel_pg_ests  0 \
 --feat_rel_pg_ests_onehot  0 \
 --feat_pg_est_one_hot  0 \
 --flow_features 1 --feat_tolerance 0 \
 --lr 0.0001"
echo $CMD
eval $CMD


CMD="time python3 main.py --algs nn -n -1 \
 --max_discrete_featurizing_buckets 10 \
 --hidden_layer_size 512 \
 --weight_decay 0.1 \
 --alg $ALG \
 --loss_func $LOSS_FUNC \
 --nn_type $NN_TYPE \
 --sample_bitmap $SAMPLE_BITMAP \
 --test_size 0.5 \
 --exp_prefix final_runs \
 --result_dir \
  all_results/inl_fixed_scan_ops/nested_loop_index7/default/ablation \
 --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
 --eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
 --optimizer_name adamw \
 --normalize_flow_loss 1 \
 --eval_on_job 1 \
 --join_features 1 \
 --pred_features 0 \
 --heuristic_features 1 \
 --feat_rel_pg_ests  0 \
 --feat_rel_pg_ests_onehot  0 \
 --feat_pg_est_one_hot  0 \
 --flow_features 1 --feat_tolerance 0 \
 --lr 0.0001"
echo $CMD
eval $CMD

CMD="time python3 main.py --algs nn -n -1 \
 --max_discrete_featurizing_buckets 10 \
 --hidden_layer_size 512 \
 --weight_decay 0.1 \
 --alg $ALG \
 --loss_func $LOSS_FUNC \
 --nn_type $NN_TYPE \
 --sample_bitmap $SAMPLE_BITMAP \
 --test_size 0.5 \
 --exp_prefix final_runs \
 --result_dir \
  all_results/inl_fixed_scan_ops/nested_loop_index7/default/ablation \
 --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
 --eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
 --optimizer_name adamw \
 --normalize_flow_loss 1 \
 --eval_on_job 1 \
 --join_features 1 \
 --pred_features 1 \
 --table_features 0 \
 --heuristic_features 1 \
 --feat_rel_pg_ests  0 \
 --feat_rel_pg_ests_onehot  0 \
 --feat_pg_est_one_hot  0 \
 --flow_features 1 --feat_tolerance 0 \
 --lr 0.0001"
echo $CMD
eval $CMD

CMD="time python3 main.py --algs nn -n -1 \
 --max_discrete_featurizing_buckets 10 \
 --hidden_layer_size 512 \
 --weight_decay 0.1 \
 --alg $ALG \
 --loss_func $LOSS_FUNC \
 --nn_type $NN_TYPE \
 --sample_bitmap $SAMPLE_BITMAP \
 --test_size 0.5 \
 --exp_prefix final_runs \
 --result_dir \
  all_results/inl_fixed_scan_ops/nested_loop_index7/default/ablation \
 --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
 --eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
 --optimizer_name adamw \
 --normalize_flow_loss 1 \
 --eval_on_job 1 \
 --join_features 0 \
 --pred_features 0 \
 --table_features 0 \
 --heuristic_features 1 \
 --feat_rel_pg_ests  1 \
 --feat_rel_pg_ests_onehot  1 \
 --feat_pg_est_one_hot  1 \
 --flow_features 1 --feat_tolerance 0 \
 --lr 0.0001"
echo $CMD
eval $CMD

CMD="time python3 main.py --algs nn -n -1 \
 --max_discrete_featurizing_buckets 10 \
 --hidden_layer_size 512 \
 --weight_decay 0.1 \
 --alg nn \
 --loss_func mse \
 --nn_type mscn \
 --sample_bitmap 1 \
 --test_size 0.5 \
 --exp_prefix final_runs \
 --result_dir \
  all_results/inl_fixed_scan_ops/nested_loop_index7/default/bitmap_new \
 --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
 --eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
 --optimizer_name adamw \
 --normalize_flow_loss 1 \
 --eval_on_job 1 \
 --feat_rel_pg_ests  1 \
 --feat_rel_pg_ests_onehot  1 \
 --feat_pg_est_one_hot  1 \
 --flow_features 1 --feat_tolerance 0 \
 --lr 0.0001"
echo $CMD
eval $CMD

CMD="time python3 main.py --algs nn -n -1 \
 --max_discrete_featurizing_buckets 1 \
 --hidden_layer_size 512 \
 --weight_decay 0.1 \
 --alg nn \
 --loss_func mse \
 --nn_type mscn \
 --sample_bitmap 1 \
 --test_size 0.5 \
 --exp_prefix final_runs \
 --result_dir \
  all_results/inl_fixed_scan_ops/nested_loop_index7/default/bitmap_new \
 --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
 --eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
 --optimizer_name adamw \
 --normalize_flow_loss 1 \
 --eval_on_job 1 \
 --feat_rel_pg_ests  1 \
 --feat_rel_pg_ests_onehot  1 \
 --feat_pg_est_one_hot  1 \
 --flow_features 1 --feat_tolerance 0 \
 --lr 0.0001"
echo $CMD
eval $CMD
