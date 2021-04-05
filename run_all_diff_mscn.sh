LOSS_FUNC=$1
DECAY=$2
LR=$3
COST_MODEL=$4
NORM_FLOW_LOSS=$5

DEBUG_SET=0
DEBUG_RATIO=10.0
MAX_EPOCHS=10

ALG=nn
NN_TYPE=mscn_set

QUERY_MB_SIZE=4
EVAL_ON_JOB=0
EVAL_ON_JOBM=0
PRIORITY=0.0
PR_NORM=no
SAMPLE_BITMAP_BUCKETS=1000
SAMPLE_BITMAP=0

PRELOAD_FEATURES=1
USE_SET_PADDING=2

NUM_MSE_ANCHORING=-1

FLOW_FEATS=1
SWITCH_EPOCH=100000
REL_ESTS=1
ONEHOT=1

USE_VAL_SET=0

SEEDS=(6 7 8)

EVAL_EPOCH=100

LOSSES=mysql-loss,qerr
#LOSSES=qerr

NHL=2
RES_DIR=all_results/mysql/fcnn/diff/final_rc4

BUCKETS=10
HLS=256

LOAD_QUERY_TOGTHER=0
NUM_PAR=16

for i in "${!SEEDS[@]}";
  do
  CMD="time python3 main.py --algs nn -n -1 \
   --hidden_layer_size $HLS \
   --debug_set $DEBUG_SET \
   --debug_ratio $DEBUG_RATIO \
   --use_val_set $USE_VAL_SET \
   --query_mb_size $QUERY_MB_SIZE \
   --test_diff_templates 1 \
   --diff_templates_seed ${SEEDS[$i]} \
   --num_mse_anchoring $NUM_MSE_ANCHORING \
   --num_hidden_layers $NHL \
   --max_discrete_featurizing_buckets $BUCKETS \
   --sampling_priority_alpha $PRIORITY \
   --priority_normalize_type $PR_NORM \
   --weight_decay $DECAY \
   --alg $ALG \
   --load_query_together $LOAD_QUERY_TOGTHER \
   --job_skip_zero_queries 0 \
   --losses $LOSSES
   --loss_func $LOSS_FUNC \
   --nn_type $NN_TYPE \
   --sample_bitmap $SAMPLE_BITMAP \
   --sample_bitmap_buckets $SAMPLE_BITMAP_BUCKETS \
   --preload_features $PRELOAD_FEATURES \
   --use_set_padding $USE_SET_PADDING \
   --test_size 0.5 \
   --exp_prefix final_runs \
   --result_dir $RES_DIR \
   --max_epochs $MAX_EPOCHS --cost_model $COST_MODEL \
   --switch_loss_fn_epoch $SWITCH_EPOCH \
   --eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
   --optimizer_name adamw \
   --normalize_flow_loss $NORM_FLOW_LOSS \
   --eval_on_job 0 \
   --eval_on_jobm 0 \
   --add_job_features 0 \
   --feat_rel_pg_ests  $REL_ESTS \
   --feat_rel_pg_ests_onehot  $ONEHOT \
   --feat_pg_est_one_hot  $ONEHOT \
   --flow_features $FLOW_FEATS --feat_tolerance 0 \
   --lr $LR"
    echo $CMD
    eval $CMD
done
