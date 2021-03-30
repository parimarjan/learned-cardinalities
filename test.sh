time python3 main.py --algs nn --debug_set 1 --eval_epoch 2 \
--user root --pwd password \
--loss_func flow_loss2 --nn_type microsoft \
--query_template all \
--eval_epoch_qerr 2 \
--eval_epoch_jerr 100 \
--eval_epoch_plan_err 2 --eval_epoch_flow_err 2 --exp_prefix \
normFL-slowerLR-gradClip --cost_model mysql_rc4 --use_val_set 0 \
--flow_features 1 \
--max_epochs 10 --lr 0.000001 --hidden_layer_size 512 \
--num_hidden_layers 4 \
--weight_decay 1.0 --normalize_flow_loss 1 \
--result_dir mysqlrc4_debug \
--eval_test_while_training 0 \
--losses qerr --debug_ratio 2 --clip_gradient 2.0

time python3 main.py --algs nn --debug_set 1 --eval_epoch 2 \
--user root --pwd password \
--loss_func flow_loss2 --nn_type microsoft \
--query_template all \
--eval_epoch_qerr 2 \
--eval_epoch_jerr 100 \
--eval_epoch_plan_err 2 --eval_epoch_flow_err 2 --exp_prefix \
noNorm-slowerLR-gradClip --cost_model mysql_rc4 --use_val_set 0 \
--flow_features 1 \
--max_epochs 10 --lr 0.000001 --hidden_layer_size 512 \
--num_hidden_layers 4 \
--weight_decay 1.0 --normalize_flow_loss 0 \
--result_dir mysqlrc4_debug \
--eval_test_while_training 0 \
--losses qerr --debug_ratio 2 --clip_gradient 2.0

time python3 main.py --algs nn --debug_set 1 --eval_epoch 2 \
--user root --pwd password \
--loss_func flow_loss2 --nn_type microsoft \
--query_template all \
--eval_epoch_qerr 2 \
--eval_epoch_jerr 100 \
--eval_epoch_plan_err 2 --eval_epoch_flow_err 2 --exp_prefix \
noNorm-evenSlowerLR --cost_model mysql_rc4 --use_val_set 0 \
--flow_features 1 \
--max_epochs 10 --lr 0.0000001 --hidden_layer_size 512 \
--num_hidden_layers 4 \
--weight_decay 1.0 --normalize_flow_loss 0 \
--result_dir mysqlrc4_debug \
--eval_test_while_training 0 \
--losses qerr --debug_ratio 2
