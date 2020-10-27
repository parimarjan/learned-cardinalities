NN_TYPE=$1
LOSS_FUNC=$2
TEST_SIZE=$3
EPOCHS=$4
HLS=128
NH=2

python3 main.py --algs nn --nn_type $NN_TYPE --query_directory job_queries \
--eval_on_job 1 --eval_on_jobm 1 --test_size $TEST_SIZE --skip_zero_queries 0 \
--num_hidden_layers $NH --hidden_layer_size $HLS \
--weight_decay 1.0 \
--lr 0.001 --max_epochs $EPOCHS --eval_epoch 5 \
--eval_epoch_plan_err 100 --eval_epoch_flow_err 100 \
--eval_epoch_jerr 100 --loss_func $LOSS_FUNC
