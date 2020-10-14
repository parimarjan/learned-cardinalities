#DIFF_SEEDS=(1 2 3 4 5)
#DIFF_SEEDS=(4 5 1 2 3)
#DIFF_SEEDS=(4 5 2 3)
#MIN_QERRS=(1.0)
#DECAYS=(0.1)
#DIFF_SEEDS=(2 3)
MAX_EPOCHS=20

ALG=$1
LOSS_FUNC=$2
NN_TYPE=$3

BUCKETS=(10)
#HLS=(64 128 256)
HLS=(512)
FLOW_FEATS=1
EVAL_EPOCH=100
SAMPLE_BITMAP=0

for i in "${!BUCKETS[@]}";
  do
  for j in "${!HLS[@]}";
    do
    CMD="time python3 main.py --algs nn -n -1 \
     --max_discrete_featurizing_buckets ${BUCKETS[$i]} \
     --hidden_layer_size ${HLS[$j]} \
     --alg $ALG \
     --loss_func $LOSS_FUNC \
     --nn_type $NN_TYPE \
     --sample_bitmap $SAMPLE_BITMAP \
     --test_size 0.5 \
     --exp_prefix final_runs \
     --result_dir \
      all_results/inl_fixed_scan_ops/nested_loop_index7/default/final_results_baselines \
     --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
     --eval_epoch $EVAL_EPOCH --join_loss_pool_num 60 \
     --optimizer_name adamw \
     --normalize_flow_loss 1 \
     --weight_decay 0.1 \
     --eval_on_job 1 \
     --feat_rel_pg_ests  1 \
     --feat_rel_pg_ests_onehot  1 \
     --feat_pg_est_one_hot  1 \
     --flow_features $FLOW_FEATS --feat_tolerance 0 \
     --lr 0.0001"
    echo $CMD
    eval $CMD
  done
done
