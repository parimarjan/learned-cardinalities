ALG=$1
LOSS_FUNC=$2
LR=$3
MAX_EPOCHS=$4

QUERY_DIR=our_dataset/queries
TRAIN_CARD_KEY=wanderjoin
SAMPLING_KEY=wanderjoin
PRIORITY=0.0
NN_TYPE=microsoft
SAMPLE_BITMAP=0
LOSSES=join-loss,qerr

HLS=512
NHL=4
RES_DIR=/flash1/pari/VLDB-Nov1-Results/all_results/vldb/wanderjoin2/

LOAD_QUERY_TOGTHER=0
BUCKETS=10
DECAY=1.0
NUM_PAR=40

FLOW_FEATS=1
EVAL_EPOCH=40

REPS=3
for i in {0..$REPS}
do
CMD="time python3 main.py --algs nn -n -1 \
 --use_val_set 1 \
 --hidden_layer_size $HLS \
 --num_hidden_layers $NHL \
 --query_directory $QUERY_DIR \
 --train_card_key $TRAIN_CARD_KEY \
 --sampling_key $SAMPLING_KEY \
 --max_discrete_featurizing_buckets $BUCKETS \
 --sampling_priority_alpha $PRIORITY \
 --weight_decay $DECAY \
 --alg $ALG \
 --load_query_together $LOAD_QUERY_TOGTHER \
 --eval_on_job 0 \
 --losses $LOSSES
 --loss_func $LOSS_FUNC \
 --nn_type $NN_TYPE \
 --sample_bitmap $SAMPLE_BITMAP \
 --test_size 0.5 \
 --exp_prefix final_runs \
 --result_dir $RES_DIR \
 --max_epochs $MAX_EPOCHS --cost_model nested_loop_index7 \
 --eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
 --optimizer_name adamw \
 --normalize_flow_loss 1 \
 --feat_rel_pg_ests  1 \
 --feat_rel_pg_ests_onehot  1 \
 --feat_pg_est_one_hot  1 \
 --flow_features $FLOW_FEATS --feat_tolerance 0 \
 --lr $LR"
echo $CMD
eval $CMD
done
