python3 main.py --algs nn --nn_type microsoft \
--db_year_train 1950 \
--db_year_test 1950,1960,1970,1980,1990,2000 \
--query_template 1a,2a,3a,4a,9a,9b,10a,11a,6a \
--num_hidden_layers 4 --hidden_layer_size 256 \
--loss_func flow_loss2 --normalize_flow_loss 1 \
--max_epochs 15 --lr 0.00005 --eval_epoch 100 \
--query_mb_size 4 --flow_features 0
