#ALG=nn
ALG=$1
LOSS_FUNC=$2
NN_TYPE=$3
LR=$4
HLS=$5

REPS=5

PRIORITY=0.0
FL_NORM=0

NH=4
MAX_EPOCHS=30

REL=1
ONEHOT=1
USE_VAL_SET=1

TEST_SIZE=0.5
BUCKETS=10
#EVAL_EPOCH=$MAX_EPOCHS
EVAL_EPOCH=1
RES_DIR=./all_results/vldb/so/default/multiruns_lc
#RES_DIR=./so_fcnn_hl2/

FLOW_FEATS=1
DECAY=1.0

for (( c=0; c<$REPS; c++ ))
do
CMD="time python3 main.py --algs $ALG -n -1 \
 --loss_func $LOSS_FUNC \
 --normalize_flow_loss $FL_NORM \
 --lr $LR \
 --use_val_set $USE_VAL_SET \
 --weight_decay $DECAY \
 --test_size $TEST_SIZE \
 --join_loss_pool_num 10 \
 --db_name so --query_dir so_workload \
 --nn_type $NN_TYPE \
 --sampling_priority_alpha $PRIORITY \
 --exp_prefix final_default \
 --result_dir $RES_DIR \
 --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
 --eval_epoch $EVAL_EPOCH \
 --eval_epoch_jerr $EVAL_EPOCH --eval_epoch_flow_err $EVAL_EPOCH \
 --eval_epoch_plan_err 1000 \
 --hidden_layer_size $HLS --optimizer_name adamw \
 --num_hidden_layers $NH \
 --test_diff_templates 0 \
 --feat_rel_pg_ests  $REL \
 --feat_rel_pg_ests_onehot  $ONEHOT \
 --feat_pg_est_one_hot  $ONEHOT \
 --flow_features $FLOW_FEATS --feat_tolerance 0 \
 --max_discrete_featurizing_buckets $BUCKETS"
echo $CMD
eval $CMD
done
