MIN_QERRS=(1.0)
DECAY=1.0
DIFF_SEEDS=(10 9 8 7 6 1 2 3 4 5)

#DIFF_SEEDS=(6)

PRIORITY=0.5
MAX_EPOCHS=(10)
#MAX_EPOCHS=(10)
BUCKETS=10
FLOW_FEATS=1
LR=0.0001
PRELOAD_FEATURES=1
No7=0
RES_DIR=all_results/vldb/test_diff/xgb/
LOSSES=join-loss,qerr

REL_ESTS=1
ONEHOT=1
MB_SIZE=4

DEPTH=$1
NORM_FLOW_LOSS=0
NUM_WORKERS=0

HLS=256
NUM_HLS=2
LOAD_QUERY_TOGETHER=0

WEIGHTED_MSES=(0.0)
NUM_MSE_ANCHORING=(-1)

JOB_FEATS=1
TEST_FEATS=1

SAMPLE_BITMAP=0
SAMPLE_BITMAP_BUCKETS=1000
EVAL_EPOCH=500
EVAL_ON_JOB=0
USE_SET_PADDING=3

for i in "${!WEIGHTED_MSES[@]}";
do
  #for j in "${!NUM_HLS[@]}";
  for j in "${!MAX_EPOCHS[@]}";
  do
  for k in "${!DIFF_SEEDS[@]}";
    do
    CMD="time python3 main.py --algs xgboost -n -1 \
     --losses $LOSSES \
     --result_dir $RES_DIR \
     --max_depth $DEPTH \
     --cost_model nested_loop_index7 \
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
     --max_discrete_featurizing_buckets $BUCKETS --lr $LR"
    echo $CMD
    eval $CMD
    done
  done
done
