ALG=nn
PRIORITY=0.0
LOSS_FUNC=$1
LR=$2
TEST_SIZE=$3
HLS=$4
BUCKETS=$5
MAX_EPOCHS=$6
NN_TYPE=$7
RES_DIR=$8
FLOW_FEATS=0
DECAY=$9
DIFF_SEEDS=(1 2 3 4 5 6 7 8 9 10)
#MAX_EPOCHS=10
EVAL_EPOCH=$MAX_EPOCHS
#GRAD_CLIP=10.0
REL=0
ONEHOT=0

#python3 main.py --algs nn --db_name so --query_dir so_workload/
#--query_template 1,11,17,18,18b --test_diff 1 --diff_templates_type 3
#--diff_templates_seed 1 --max_epochs 50 --flow_features 0 -n -1 --eval_epoch 20
#--loss_func mse --lr 0.001 --test_size 0.3 --diff_templates_seed 2

for k in "${!DIFF_SEEDS[@]}";
	do
	CMD="time python3 main.py --algs $ALG -n -1 \
	 --loss_func $LOSS_FUNC \
	 --lr $LR \
   --weight_decay $DECAY \
   --test_size $TEST_SIZE \
   --join_loss_pool_num 10 \
   --db_name so --query_dir so_workload \
	 --nn_type $NN_TYPE \
	 --sampling_priority_alpha $PRIORITY \
	 --exp_prefix diff_partitions \
	 --result_dir $RES_DIR \
	 --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
	 --eval_epoch $EVAL_EPOCH \
	 --eval_epoch_jerr $EVAL_EPOCH --eval_epoch_flow_err $EVAL_EPOCH \
	 --eval_epoch_plan_err 1000 \
	 --hidden_layer_size $HLS --optimizer_name adamw \
	 --test_diff_templates 1 --diff_templates_type 3 \
	 --diff_templates_seed ${DIFF_SEEDS[$k]} \
	 --feat_rel_pg_ests  $REL \
	 --feat_rel_pg_ests_onehot  $ONEHOT \
	 --feat_pg_est_one_hot  $ONEHOT \
	 --flow_features $FLOW_FEATS --feat_tolerance 0 \
	 --max_discrete_featurizing_buckets $BUCKETS"
	echo $CMD
	eval $CMD
done
