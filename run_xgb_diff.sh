DIFF_SEEDS=(3 8 9 10 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40)
#DIFF_SEEDS=(11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
#HLS=512
#ALG=$1
#LOSS_FUNC=$2
#NN_TYPE=$3

JOB_FEATS=1
TEST_FEATS=1
BUCKETS=10
SAMPLE_BITMAP=0
SAMPLE_BITMAP_BUCKETS=1000
EVAL_EPOCH=40
NORM_FLOW_LOSS=1
NUM_PAR=40
FLOW_FEATS=1
ALG=xgboost

#python3 main.py --algs xgboost --eval_on_job 1
#--max_discrete_featurizing_buckets 10 --debug_set 0 --test_diff 1
#--diff_templates_seed 6 --result_dir all_results/xgboost/test_diff/^C

for k in "${!DIFF_SEEDS[@]}";
  do
  CMD="time python3 main.py --algs $ALG -n -1 \
   --exp_prefix xgb_diff_partitions \
   --result_dir \
    all_results/xgboost/test_diff/ \
   --cost_model nested_loop_index7 \
   --eval_epoch 40 --join_loss_pool_num $NUM_PAR \
   --test_diff_templates 1 --diff_templates_type 3 \
   --diff_templates_seed ${DIFF_SEEDS[$k]} \
   --sample_bitmap $SAMPLE_BITMAP \
   --sample_bitmap_num 1000 \
   --sample_bitmap_buckets $SAMPLE_BITMAP_BUCKETS \
   --eval_on_job 1 \
   --feat_rel_pg_ests  1 \
   --feat_rel_pg_ests_onehot  1 \
   --feat_pg_est_one_hot  1 \
   --flow_features $FLOW_FEATS --feat_tolerance 0 \
   --max_discrete_featurizing_buckets $BUCKETS"
  echo $CMD
  eval $CMD
done
