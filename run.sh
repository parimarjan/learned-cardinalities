
time python3 main.py --algs nn --loss_func flow_loss2 --losses \
qerr,mysql-loss,flow-loss --hidden_layer_size 512 --num_hidden_layers 4 \
--debug_set 0 --debug_ratio 1 \
--query_template all --cost_model mysql_rc --user root --pwd password \
--eval_epoch_qerr 2 -n -1 --query_dir queries/imdb/ --flow_features 1 \
--nn_type microsoft --test_diff_templates 1 \
--diff_templates_seed 7 --lr 0.0001 \
--result_dir all_results/mysql/diff_templates1 \
--weight_decay 1.0 --max_epochs 8 --normalize_flow_loss 1

