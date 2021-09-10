LOSS_FUNC=$1
DECAY=$2
LR=$3
ONEHOT=$4
FLOW_FEATS=$5

# TODO: add job features
EVAL_ON_JOB=1

ALG=nn
NN_TYPE=mscn_set

REL_ESTS=1

NORM_FLOW_LOSS=0

MIN_QERRS=(1.0)

# battleground seeds
DIFF_SEEDS=(2 5 6 7 8 9 10 1 2 4 11)
#DIFF_SEEDS=(7 13)
#DIFF_SEEDS=(16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
EVAL_EPOCH=100

#DIFF_SEEDS=(13)

#DIFF_SEEDS=(6)

PRIORITY=0.0
MAX_EPOCHS=(10)
#MAX_EPOCHS=(10)
BUCKETS=10
LR=0.0001
#LR=0.0001
PRELOAD_FEATURES=1
No7=0
#RES_DIR=all_results/vldb/test_diff/fcnn/wd1
#RES_DIR=/flash1/pari/VLDB-Nov1-Results/all_results/vldb/test_diff/mscn/lc
RES_DIR=/flash1/pari/VLDB-Nov1-Results/all_results/vldb/test_diff/mscn/finalv2

MB_SIZE=4

NUM_WORKERS=0

HLS=256
NUM_HLS=2
LOAD_QUERY_TOGETHER=0

WEIGHTED_MSES=(0.0)
NUM_MSE_ANCHORING=(-1)

JOB_FEATS=0
TEST_FEATS=1

SAMPLE_BITMAP=0
SAMPLE_BITMAP_BUCKETS=1000
USE_SET_PADDING=3

for i in "${!WEIGHTED_MSES[@]}";
do
  #for j in "${!NUM_HLS[@]}";
  for j in "${!MAX_EPOCHS[@]}";
  do
  for k in "${!DIFF_SEEDS[@]}";
    do
    CMD="time python3 main.py --algs $ALG -n -1 \
     --loss_func $LOSS_FUNC \
     --losses qerr,join-loss \
     --use_set_padding $USE_SET_PADDING \
     --query_mb_size $MB_SIZE \
     --no7a $No7 \
     --nn_type $NN_TYPE \
     --num_workers $NUM_WORKERS \
     --load_query_together $LOAD_QUERY_TOGETHER \
     --sampling_priority_alpha $PRIORITY \
     --preload_features $PRELOAD_FEATURES \
     --add_job_features $JOB_FEATS \
     --add_test_features $TEST_FEATS \
     --weighted_mse ${WEIGHTED_MSES[$i]} \
     --weight_decay $DECAY \
     --exp_prefix runAllDiff \
     --result_dir $RES_DIR \
     --max_epochs ${MAX_EPOCHS[$j]} \
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
     --eval_on_jobm $EVAL_ON_JOB \
     --add_job_features $EVAL_ON_JOB \
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
