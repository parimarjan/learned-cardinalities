ALG=nn
PRIORITY=0.0
LOSS_FUNC=$1
NN_TYPE=$2
LR=$3
REL=0
ONEHOT=0

TEST_SIZE=0.5
HLS=128
BUCKETS=25
MAX_EPOCHS=50
#EVAL_EPOCH=$MAX_EPOCHS
EVAL_EPOCH=2
RES_DIR=./vldb_results/so/default/lcs
FLOW_FEATS=1
DECAY=1.0
#DIFF_SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
#MAX_EPOCHS=10
#GRAD_CLIP=10.0

CMD="time python3 main.py --algs $ALG -n -1 \
 --loss_func $LOSS_FUNC \
 --lr $LR \
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
 --test_diff_templates 0 \
 --feat_rel_pg_ests  $REL \
 --feat_rel_pg_ests_onehot  $ONEHOT \
 --feat_pg_est_one_hot  $ONEHOT \
 --flow_features $FLOW_FEATS --feat_tolerance 0 \
 --max_discrete_featurizing_buckets $BUCKETS"
echo $CMD
eval $CMD
