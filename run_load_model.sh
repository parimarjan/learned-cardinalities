#DIFF_SEEDS=(1 2 3 4 5)
#DIFF_SEEDS=(4 5 1 2 3)
#DIFF_SEEDS=(4 5 2 3)
#MIN_QERRS=(1.0)
#DECAYS=(1.0 0.1)
#DECAYS=(20.0 100.0)
#NUM_HLS=(2)
#DIFF_SEEDS=(1 2 3 4 5)
#DIFF_SEEDS=(2)
#DIFF_SEEDS=(1)
MAX_EPOCHS=0
HLS=128
#ALG=$1
#LOSS_FUNC=$2
SAMPLE_BITMAP=0

# 8
#MODEL_DIR=/home/pari/learned-cardinalities/all_results/vldb/test_diff/mscn/run1/runAllDiff-nested_loop_index7-NN-df:10-nn:2:128-loss:mse-0.0--D1.0-333/

# 6
MODEL_DIR=/home/pari/learned-cardinalities/all_results/vldb/test_diff/mscn/run1/runAllDiff-nested_loop_index7-NN-df:10-nn:2:128-loss:mse-0.0--D1.0-440/


#CMD="python3 main.py --algs nn \
#--query_template 1a -n 2 \
#--model_dir $MODEL_DIR \
#--eval_on_job 0 \
#--max_epochs 0 --eval_epoch 20"

CMD="time python3 main.py --algs nn -n -1 \
 --debug_set 0 \
 --loss_func mse \
 --nn_type mscn_set \
 --exp_prefix load_model \
 --result_dir \
  all_results/vldb/test_diff/mscn/run1_load \
 --max_epochs 0 --cost_model nested_loop_index7 \
 --eval_epoch 100 --join_loss_pool_num 10 \
 --losses qerr,join-loss \
 --hidden_layer_size $HLS --optimizer_name adamw \
 --num_hidden_layers 2
 --normalize_flow_loss 1 \
 --test_diff_templates 1 --diff_templates_type 3 \
 --diff_templates_seed 6 \
 --sample_bitmap $SAMPLE_BITMAP \
 --sample_bitmap_num 1000 \
 --sample_bitmap_buckets 1000 \
 --min_qerr 1.00 \
 --weight_decay 1.0 \
 --eval_on_job 1 \
 --job_skip_zero_queries 0 \
 --feat_rel_pg_ests  1 \
 --feat_rel_pg_ests_onehot  1 \
 --feat_pg_est_one_hot  1 \
 --flow_features 0 --feat_tolerance 0 \
 --model_dir $MODEL_DIR
 --max_discrete_featurizing_buckets 10 --lr 0.0001"
echo $CMD
eval $CMD
