MIN_QERRS=(1.0)
DECAY=1.0
#DIFF_SEEDS=(2 6 7 8 1 3 4 5 9 10)
#DIFF_SEEDS=(2 6 7 8 9 10 1 3 5)
#DIFF_SEEDS=(6 7 8 9 10 1 3 5)
DIFF_SEEDS=(9 10 1 3 5)
#DIFF_SEEDS=(6 8 2 7)
PRIORITY=0.0
MAX_EPOCHS=10
BUCKETS=10
FLOW_FEATS=0
LR=0.0001
PRELOAD_FEATURES=1

ALG=$1
LOSS_FUNC=$2
NN_TYPE=$3
NORM_FLOW_LOSS=1
NUM_WORKERS=5

HLS=128
NUM_HLS=2
LOAD_QUERY_TOGETHER=0

#WEIGHTED_MSES=(0.005 0.0025 0.0075)
WEIGHTED_MSES=(0.0)
NUM_MSE_ANCHORING=(-1)

#WEIGHTED_MSES=(0.0)
#WEIGHTED_MSES=(0.01)
#NUM_MSE_ANCHORING=(10 50 100)
#NUM_MSE_ANCHORING=(-1 10)

JOB_FEATS=1
TEST_FEATS=1

SAMPLE_BITMAP=0
SAMPLE_BITMAP_BUCKETS=1000
EVAL_EPOCH=500
EVAL_ON_JOB=1
NUM_PAR=20

for i in "${!WEIGHTED_MSES[@]}";
do
  #for j in "${!NUM_HLS[@]}";
  for j in "${!NUM_MSE_ANCHORING[@]}";
  do
  for k in "${!DIFF_SEEDS[@]}";
    do
    CMD="time python3 main.py --algs $ALG -n -1 \
     --loss_func $LOSS_FUNC \
     --nn_type $NN_TYPE \
     --num_workers $NUM_WORKERS \
     --load_query_together $LOAD_QUERY_TOGETHER \
     --sampling_priority_alpha $PRIORITY \
     --preload_features $PRELOAD_FEATURES \
     --add_job_features $JOB_FEATS \
     --add_test_features $TEST_FEATS \
     --weighted_mse ${WEIGHTED_MSES[$i]} \
     --num_mse_anchoring ${NUM_MSE_ANCHORING[$j]} \
     --weight_decay $DECAY \
     --exp_prefix runAllDiff \
     --result_dir all_results/vldb/test_diff/mscn/run1
     --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
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
     --eval_on_job $EVAL_ON_JOB \
     --feat_rel_pg_ests  1 \
     --feat_rel_pg_ests_onehot  1 \
     --feat_pg_est_one_hot  1 \
     --flow_features $FLOW_FEATS --feat_tolerance 0 \
     --max_discrete_featurizing_buckets $BUCKETS --lr $LR"
    echo $CMD
    eval $CMD
    done
  done
done
