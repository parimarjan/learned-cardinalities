ALG=$1
LOSS_FUNC=$2
TRAIN_CARD_KEY=$3
SAMPLING_KEY=$4
PRIORITY=$5
NN_TYPE=mscn
SAMPLE_BITMAP=0
MAX_EPOCHS=10
LOSSES=join-loss,qerr

HLS=512
NHL=2
RES_DIR=all_results/inl_fixed_scan_ops/nested_loop_index7/wj/results1

LOAD_QUERY_TOGTHER=0
BUCKETS=10
DECAY=0.1
NUM_PAR=40

FLOW_FEATS=1
EVAL_EPOCH=40

#python3 main.py --algs nn --sampling_key wanderjoin -n 10 --eval_on_job 0
#--train_card_key wanderjoin

CMD="time python3 main.py --algs nn -n -1 \
 --hidden_layer_size $HLS \
 --num_hidden_layers $NHL \
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
 --lr 0.0001"
echo $CMD
eval $CMD
