
#time python3 main.py --algs nn --debug_set 1 --eval_epoch 2 --loss_func \
#flow_loss2 --nn_type microsoft --query_template all --eval_epoch_qerr 2 \
#--eval_epoch_jerr 100 \
#--eval_epoch_plan_err 2 --eval_epoch_flow_err 2 --exp_prefix \
#noNorm --cost_model mysql_rc4 --use_val_set 0 \
#--flow_features 1 \
#--max_epochs 20 --lr 0.0001 --hidden_layer_size 512 \
#--num_hidden_layers 4 \
#--weight_decay 0.1 --normalize_flow_loss 0 \
#--losses mysql-loss,qerr --debug_ratio 2

#time python3 main.py --algs nn --debug_set 1 --eval_epoch 2 --loss_func \
#flow_loss2 --nn_type microsoft --query_template all --eval_epoch_qerr 2 \
#--eval_epoch_jerr 100 \
#--eval_epoch_plan_err 2 --eval_epoch_flow_err 2 --exp_prefix \
#nonormslow --cost_model mysql_rc4 --use_val_set 0 \
#--flow_features 1 \
#--max_epochs 20 --lr 0.00001 --hidden_layer_size 512 \
#--num_hidden_layers 4 \
#--weight_decay 0.1 --normalize_flow_loss 0 \
#--losses mysql-loss,qerr --debug_ratio 2


time python3 main.py --algs nn --debug_set 1 --eval_epoch 2 --loss_func \
flow_loss2 --nn_type microsoft --query_template all --eval_epoch_qerr 2 \
--eval_epoch_jerr 100 \
--eval_epoch_plan_err 2 --eval_epoch_flow_err 2 --exp_prefix \
normFL-Slow-smallerConstants --cost_model mysql_rc4 --use_val_set 0 \
--flow_features 1 \
--max_epochs 20 --lr 0.00001 --hidden_layer_size 512 \
--num_hidden_layers 4 \
--weight_decay 0.1 --normalize_flow_loss 1 \
--losses mysql-loss,qerr --debug_ratio 2

time python3 main.py --algs nn --debug_set 1 --eval_epoch 2 --loss_func \
flow_loss2 --nn_type microsoft --query_template all --eval_epoch_qerr 2 \
--eval_epoch_jerr 100 \
--eval_epoch_plan_err 2 --eval_epoch_flow_err 2 --exp_prefix \
normFL-slowerLR-smallerConstants --cost_model mysql_rc4 --use_val_set 0 \
--flow_features 1 \
--max_epochs 20 --lr 0.000001 --hidden_layer_size 512 \
--num_hidden_layers 4 \
--weight_decay 0.1 --normalize_flow_loss 1 \
--losses mysql-loss,qerr --debug_ratio 2

time python3 main.py --algs nn --debug_set 1 --eval_epoch 2 --loss_func \
flow_loss2 --nn_type microsoft --query_template all --eval_epoch_qerr 2 \
--eval_epoch_jerr 100 \
--eval_epoch_plan_err 2 --eval_epoch_flow_err 2 --exp_prefix \
normFL-fastLR-smallerConstants --cost_model mysql_rc4 --use_val_set 0 \
--flow_features 1 \
--max_epochs 20 --lr 0.0001 --hidden_layer_size 512 \
--num_hidden_layers 4 \
--weight_decay 0.1 --normalize_flow_loss 1 \
--losses mysql-loss,qerr --debug_ratio 2
