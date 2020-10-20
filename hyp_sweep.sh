ALG=$1
LOSS_FUNC=$2
NN_TYPE=$3
USE_VAL_SET=2
NUM_WORKERS=0
NO7=0
LOSSES=join-loss,qerr

#FLOW_FEATS=(1 0)
FLOW_FEATS=(1)
WEIGHTED_MSES=(0.0)
ONE_HOT_ESTS=(0)
REL_ESTS=(1 0)

DECAYS=(0.0 1.0 0.1)
#DECAYS=(1.0)
#LRS=(0.001 0.0001)
#LRS=(0.00005)
LRS=(0.0001)
#HLS=(128 256)
MAX_EPOCHS=(10)
HLS=(128)

NORM_FLOW_LOSS=(1)
NUM_MSE_ANCHORING=(-3)

PRIORITY=0.0
BUCKETS=10
DEBUG_RATIO=10

NUM_HLS=2
LOAD_QUERY_TOGETHER=0

JOB_FEATS=1
TEST_FEATS=1

SAMPLE_BITMAP=0
SAMPLE_BITMAP_BUCKETS=1000
EVAL_EPOCH=4000
NUM_PAR=10
USE_SET_PADDING=3

for i in "${!WEIGHTED_MSES[@]}";
do
  for onehot in "${!ONE_HOT_ESTS[@]}";
  do
  for rel in "${!REL_ESTS[@]}";
  do
  for j in "${!DECAYS[@]}";
  do
  for k in "${!NUM_MSE_ANCHORING[@]}";
    do
  for ff in "${!FLOW_FEATS[@]}";
    do
  for lr in "${!LRS[@]}";
    do
  for hl in "${!HLS[@]}";
    do
  for max_epoch in "${!MAX_EPOCHS[@]}";
    do
  for norm in "${!NORM_FLOW_LOSS[@]}";
    do
    CMD="time python3 main.py --algs $ALG -n -1 \
     --debug_set 0 --debug_ratio $DEBUG_RATIO \
     --no7a $NO7 \
     --losses $LOSSES \
     --use_set_padding $USE_SET_PADDING \
     --use_val_set $USE_VAL_SET \
     --loss_func $LOSS_FUNC \
     --nn_type $NN_TYPE \
     --num_workers $NUM_WORKERS \
     --load_query_together $LOAD_QUERY_TOGETHER \
     --sampling_priority_alpha $PRIORITY \
     --add_job_features $JOB_FEATS \
     --add_test_features $TEST_FEATS \
     --weighted_mse ${WEIGHTED_MSES[$i]} \
     --max_epochs ${MAX_EPOCHS[$max_epoch]} \
     --hidden_layer_size ${HLS[$hl]} \
     --cost_model nested_loop_index7 \
     --num_mse_anchoring ${NUM_MSE_ANCHORING[$k]} \
     --weight_decay ${DECAYS[$j]} \
     --normalize_flow_loss ${NORM_FLOW_LOSS[$norm]} \
     --flow_features ${FLOW_FEATS[$ff]} \
     --lr ${LRS[$lr]} \
     --feat_rel_pg_ests  ${REL_ESTS[$rel]} \
     --feat_rel_pg_ests_onehot  ${ONE_HOT_ESTS[$onehot]} \
     --feat_pg_est_one_hot ${ONE_HOT_ESTS[$onehot]} \
     --max_discrete_featurizing_buckets $BUCKETS \
     --exp_prefix default \
     --result_dir all_results/vldb/default/hyp_sweep2 \
     --eval_epoch $EVAL_EPOCH --join_loss_pool_num $NUM_PAR \
     --eval_epoch_jerr $EVAL_EPOCH --eval_epoch_flow_err $EVAL_EPOCH \
     --eval_epoch_plan_err 40 \
     --optimizer_name adamw \
     --num_hidden_layers $NUM_HLS \
     --test_diff_templates 0 --diff_templates_type 3 \
     --sample_bitmap $SAMPLE_BITMAP \
     --sample_bitmap_num 1000 \
     --sample_bitmap_buckets $SAMPLE_BITMAP_BUCKETS \
     --min_qerr 1.00 \
     --eval_on_job 0 \
     --feat_tolerance 0"
    echo $CMD
    eval $CMD
    done
    done
    done
  done
  done
  done
  done
  done
  done
done
