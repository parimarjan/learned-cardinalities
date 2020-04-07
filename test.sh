NORMALIZATION_TYPE=$1
LOSS_FUNC=$2
NN_TYPE=$3
echo $NORMALIZATION_TYPE $LOSS_FUNC $NN_TYPE

#python3 main.py --algs nn --test 1 --max_epochs 20 --train_card_key actual \
#--eval_test_while_training 1 --sampling_priority_alpha 0.0 \
#--exp_prefix baseline \
#--normalization_type $NORMALIZATION_TYPE \
#--loss_func $LOSS_FUNC \
#--nn_type $NN_TYPE \
#--join_loss_pool_num 40

#python3 main.py --algs nn --test 1 --max_epochs 20 --train_card_key actual \
#--eval_test_while_training 1 --sampling_priority_alpha 2.0 \
#--exp_prefix baselinePriority \
#--normalization_type $NORMALIZATION_TYPE \
#--loss_func $LOSS_FUNC \
#--nn_type $NN_TYPE \
#--join_loss_pool_num 40

python3 main.py --algs nn --test 1 --max_epochs 20 --train_card_key wanderjoin \
--eval_test_while_training 1 --sampling_priority_alpha 2.0 \
--exp_prefix wj \
--normalization_type $NORMALIZATION_TYPE \
--loss_func $LOSS_FUNC \
--nn_type $NN_TYPE \
--join_loss_pool_num 40

python3 main.py --algs nn --test 1 --max_epochs 20 --train_card_key wanderjoin0.5 \
--eval_test_while_training 1 --sampling_priority_alpha 2.0 \
--exp_prefix wj0.5 \
--normalization_type $NORMALIZATION_TYPE \
--loss_func $LOSS_FUNC \
--nn_type $NN_TYPE \
--join_loss_pool_num 40

python3 main.py --algs nn --test 1 --max_epochs 50 --train_card_key actual \
--eval_test_while_training 1 --sampling_priority_alpha 2.0 \
--exp_prefix divideLenPriority --priority_err_divide_len 1 \
--normalization_type $NORMALIZATION_TYPE \
--loss_func $LOSS_FUNC \
--nn_type $NN_TYPE \
--join_loss_pool_num 40

python3 main.py --algs nn --test 1 --max_epochs 50 --train_card_key
 actual --sampling_priority_alpha 2.0 --exp_prefix divPrAvgJL20
 --priority_err_divide_len 1 --avg_jl_num
 _last 20
