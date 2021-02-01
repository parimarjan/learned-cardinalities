python3 main.py --algs nn --nn_type microsoft --query_template $1 \
--db_year_train 1950 --db_year_test \
1950,1960,1970,1980,1990,2000 -n -1 --eval_epoch 100 \
--max_epochs $3 --loss_func $2

