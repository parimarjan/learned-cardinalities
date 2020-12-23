MIN_QERRS=(1.0)
ALG=nn
NN_TYPE=microsoft

LOSS_FUNC=$1
DECAY=$2
LR=$3
MAX_EPOCHS=$4
HEURISTIC_FEATS=$5
FLOW_FEATS=$6
TABLE_FEATS=$7
JOIN_FEATS=$8
PRED_FEATS=$9

DIFF_SEEDS=(8 9 10 1 2 3 4 5)
#DIFF_SEEDS=(2 6 7 8 10)
#DIFF_SEEDS=(6)
#DIFF_SEEDS=(1 3 4 5 9)

PRIORITY=0.0
#MAX_EPOCHS=(10)
BUCKETS=10
#FLOW_FEATS=1
PRELOAD_FEATURES=1
No7=0
#RES_DIR=all_results/vldb/test_diff/fcnn/debug
#RES_DIR=all_results/vldb/test_diff/fcnn/ablation
RES_DIR=all_results/vldb/test_diff/fcnn/run2


LOSSES=qerr,join-loss

REL_ESTS=1
ONEHOT=1
MB_SIZE=1

NORM_FLOW_LOSS=0
NUM_WORKERS=0
NUM_PAR=16

HLS=512
NUM_HLS=4
LOAD_QUERY_TOGETHER=0

WEIGHTED_MSES=(0.0)
NUM_MSE_ANCHORING=(-1)

JOB_FEATS=0
TEST_FEATS=1

SAMPLE_BITMAP=0
SAMPLE_BITMAP_BUCKETS=1000
EVAL_EPOCH=500
EVAL_ON_JOB=0
USE_SET_PADDING=3

for i in "${!WEIGHTED_MSES[@]}";
do
  for j in "${!WEIGHTED_MSES[@]}";
  #for j in "${!MAX_EPOCHS[@]}";
  do
  for k in "${!DIFF_SEEDS[@]}";
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
     --preload_features $PRELOAD_FEATURES \
     --add_job_features $JOB_FEATS \
     --add_test_features $TEST_FEATS \
     --weight_decay $DECAY \
     --lr $LR \
     --exp_prefix runAllDiff \
     --result_dir $RES_DIR \
     --max_epochs $MAX_EPOCHS \
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
     --pred_features $PRED_FEATS \
     --table_features $TABLE_FEATS \
     --heuristic_features $HEURISTIC_FEATS \
     --join_features $JOIN_FEATS \
     --max_discrete_featurizing_buckets $BUCKETS"
    echo $CMD
    eval $CMD
    done
  done
done
