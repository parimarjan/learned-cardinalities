MIN_QERRS=(1.0)
DECAY=1.0
#DIFF_SEEDS=(2 6 7 8 5 1 9 10)
#DIFF_SEEDS=(6 7 8 5 1 9 10)
DIFF_SEEDS=(10)
#DIFF_SEEDS=(7 8 5 1 9 10)
PRIORITY=0.0
MAX_EPOCHS=40
EVAL_EPOCH=2
BUCKETS=10
FLOW_FEATS=1
ALG=$1
LOSS_FUNC=$2
NN_TYPE=$3
SWITCH_EPOCH=$4
HLS=512
NUM_HLS=2

DEBUG_RATIO=5
LOAD_QUERY_TOGETHER=1
WEIGHTED_MSES=(0.0)
NNS=($NN_TYPE)
NUM_MSE_ANCHORING=(-3)

JOB_FEATS=1
TEST_FEATS=1

SAMPLE_BITMAP=0
SAMPLE_BITMAP_BUCKETS=1000
NORM_FLOW_LOSS=1
NUM_PAR=20

for i in "${!NNS[@]}";
do
  for j in "${!NUM_MSE_ANCHORING[@]}";
  do
  for k in "${!DIFF_SEEDS[@]}";
    do
    CMD="time python3 main.py --algs $ALG -n -1 \
     --debug_set 1 --debug_ratio $DEBUG_RATIO \
     --loss_func $LOSS_FUNC \
     --nn_type ${NNS[$i]} \
     --load_query_together $LOAD_QUERY_TOGETHER \
     --sampling_priority_alpha $PRIORITY \
     --add_job_features $JOB_FEATS \
     --add_test_features $TEST_FEATS \
     --weighted_mse 0.0 \
     --num_mse_anchoring ${NUM_MSE_ANCHORING[$j]} \
     --weight_decay $DECAY \
     --exp_prefix debug_set_{$DEBUG_RATIO}_seed${DIFF_SEEDS[$k]} \
     --result_dir \
      all_results/inl_fixed_scan_ops/nested_loop_index7/test_diff/switch_epoch_debugs/ \
     --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
     --switch_loss_fn_epoch $SWITCH_EPOCH \
     --eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
     --eval_epoch_jerr $EVAL_EPOCH --eval_epoch_flow_err $EVAL_EPOCH \
     --eval_epoch_plan_err 40 \
     --hidden_layer_size $HLS --optimizer_name adamw \
     --num_hidden_layers $NUM_HLS \
     --normalize_flow_loss $NORM_FLOW_LOSS \
     --test_diff_templates 1 --diff_templates_type 3 \
     --diff_templates_seed ${DIFF_SEEDS[$k]} \
     --sample_bitmap $SAMPLE_BITMAP \
     --sample_bitmap_num 1000 \
     --sample_bitmap_buckets $SAMPLE_BITMAP_BUCKETS \
     --min_qerr 1.00 \
     --eval_on_job 0 \
     --feat_rel_pg_ests  1 \
     --feat_rel_pg_ests_onehot  1 \
     --feat_pg_est_one_hot  1 \
     --flow_features $FLOW_FEATS --feat_tolerance 0 \
     --max_discrete_featurizing_buckets $BUCKETS --lr 0.0001"
    echo $CMD
    eval $CMD
    done
  done
done
