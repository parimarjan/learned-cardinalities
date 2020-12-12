MIN_QERRS=(1.0)
#DIFF_SEEDS=(1 2 3 4 5)
#DIFF_SEEDS=(1 6 8 7 9 10)
#DIFF_SEEDS=(1 2 3 4 5 6 7 8 9 10)
#DIFF_SEEDS=(7 6 8 9 10 1 2 3 4 5)
DIFF_SEEDS=(2 5 9 10 1 3 4)
#DIFF_SEEDS=(3 4 5)

#DIFF_SEEDS=(6)

ALG=$1
LOSS_FUNC=$2
NN_TYPE=$3
PRIORITY=0.5
DECAYS=(0.1)

LOSSES=qerr,join-loss
#PR_NORMS=(flow4 flow1)
PR_NORM=flow4
REP_EPOCH=1

#MAX_EPOCHS=(10)
MAX_EPOCHS=10
#MAX_EPOCHS=(10)
BUCKETS=10
FLOW_FEATS=1
#LRS=(0.00005 0.0001)
LRS=(0.0001)
PRELOAD_FEATURES=1
No7=0
RES_DIR=all_results/vldb/test_diff/fcnn/final_pr

REL_ESTS=1
ONEHOT=1
MB_SIZE=4

NORM_FLOW_LOSS=0
NUM_WORKERS=0

HLS=512
NUM_HLS=4
LOAD_QUERY_TOGETHER=0

WEIGHTED_MSE=0.0
NUM_MSE_ANCHORING=(-1)

JOB_FEATS=1
TEST_FEATS=1

SAMPLE_BITMAP=0
SAMPLE_BITMAP_BUCKETS=1000
EVAL_EPOCH=500
EVAL_ON_JOB=0
USE_SET_PADDING=2
NUM_PAR=16

  #for j in "${!NUM_HLS[@]}";
  for k in "${!DIFF_SEEDS[@]}";
    do
  for j in "${!LRS[@]}";
    do
    for i in "${!DECAYS[@]}";
    do
    CMD="time python3 main.py --algs $ALG -n -1 \
     --loss_func $LOSS_FUNC \
     --losses $LOSSES \
     --use_set_padding $USE_SET_PADDING \
     --query_mb_size $MB_SIZE \
     --no7a $No7 \
     --nn_type $NN_TYPE \
     --join_loss_pool_num $NUM_PAR \
     --num_workers $NUM_WORKERS \
     --load_query_together $LOAD_QUERY_TOGETHER \
     --sampling_priority_alpha $PRIORITY \
     --priority_normalize_type $PR_NORM \
     --reprioritize_epoch $REP_EPOCH \
     --preload_features $PRELOAD_FEATURES \
     --add_job_features $JOB_FEATS \
     --add_test_features $TEST_FEATS \
     --weight_decay ${DECAYS[$i]} \
     --exp_prefix runAllDiff \
     --result_dir $RES_DIR \
     --max_epochs $MAX_EPOCHS \
     --lr ${LRS[$j]} \
     --cost_model nested_loop_index7 \
     --eval_epoch $EVAL_EPOCH \
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
     --feat_rel_pg_ests  $REL_ESTS \
     --feat_rel_pg_ests_onehot $ONEHOT \
     --feat_pg_est_one_hot $ONEHOT \
     --flow_features $FLOW_FEATS --feat_tolerance 0 \
     --max_discrete_featurizing_buckets $BUCKETS"
    echo $CMD
    eval $CMD
    done
  done
done
