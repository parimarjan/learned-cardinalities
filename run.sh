
time python3 main.py --algs nn --loss_func flow_loss2 --losses \
qerr,mysql-loss,flow-loss --hidden_layer_size 512 --num_hidden_layers 4 \
--debug_set 1 --debug_ratio 2 \
--query_template all --cost_model mysql_rc --user root --pwd password \
--eval_epoch_qerr 2 -n -1 --query_dir queries/imdb/ --flow_features 1 \
--nn_type microsoft --test_diff_templates 1 \
--diff_templates_seed 7 --lr 0.00001 \
--result_dir all_results/mysql/diff_templates1 \
--weight_decay 1.0 --max_epochs 15 --normalize_flow_loss 1

time python3 main.py --algs nn --loss_func flow_loss2 --losses \
qerr,mysql-loss,flow-loss --hidden_layer_size 512 --num_hidden_layers 4 \
--debug_set 1 --debug_ratio 2 \
--query_template all --cost_model mysql_rc --user root --pwd password \
--eval_epoch_qerr 2 -n -1 --query_dir queries/imdb/ --flow_features 1 \
--nn_type microsoft --test_diff_templates 1 \
--diff_templates_seed 7 --lr 0.00001 \
--result_dir all_results/mysql/diff_templates1 \
--weight_decay 1.0 --max_epochs 15 --normalize_flow_loss 0

time python3 main.py --algs nn --loss_func flow_loss2 --losses \
qerr,mysql-loss,flow-loss --hidden_layer_size 512 --num_hidden_layers 4 \
--debug_set 1 --debug_ratio 2 \
--query_template all --cost_model mysql_rc --user root --pwd password \
--eval_epoch_qerr 2 -n -1 --query_dir queries/imdb/ --flow_features 1 \
--nn_type microsoft --test_diff_templates 1 \
--diff_templates_seed 7 --lr 0.00001 \
--result_dir all_results/mysql/diff_templates1 \
--weight_decay 0.1 --max_epochs 15 --normalize_flow_loss 0

time python3 main.py --algs nn --loss_func flow_loss2 --losses \
qerr,mysql-loss,flow-loss --hidden_layer_size 512 --num_hidden_layers 4 \
--debug_set 1 --debug_ratio 2 \
--query_template all --cost_model mysql_rc --user root --pwd password \
--eval_epoch_qerr 2 -n -1 --query_dir queries/imdb/ --flow_features 1 \
--nn_type microsoft --test_diff_templates 1 \
--diff_templates_seed 2 --lr 0.00001 \
--result_dir all_results/mysql/diff_templates1 \
--weight_decay 1.0 --max_epochs 15 --normalize_flow_loss 0

time python3 main.py --algs nn --loss_func flow_loss2 --losses \
qerr,mysql-loss,flow-loss --hidden_layer_size 512 --num_hidden_layers 4 \
--debug_set 1 --debug_ratio 2 \
--query_template all --cost_model mysql_rc --user root --pwd password \
--eval_epoch_qerr 2 -n -1 --query_dir queries/imdb/ --flow_features 1 \
--nn_type microsoft --test_diff_templates 1 \
--diff_templates_seed 6 --lr 0.00001 \
--result_dir all_results/mysql/diff_templates1 \
--weight_decay 1.0 --max_epochs 15 --normalize_flow_loss 0
