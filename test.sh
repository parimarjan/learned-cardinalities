#python3 main.py --db_name imdb -n -1 --algs nn --qopt_use_java 0 --test 1 \
#--result_dir hyperparameter_results --eval_epoch 2 \
#--sampling_priority_alpha 0.0 --eval_epoch_jerr 2 \
#--num_hidden_layers 1 --max_discrete_featurizing_buckets 1 \
#--hidden_layer_size 256 \
#--reprioritize_epoch 2 \
#--heuristic_features 0 --nn_type mscn --avg_jl_priority 1 \
#--eval_test_while_training 0 --lr 0.0001 --max_epochs 40 \
#--priority_err_divide_len 0 --exp_prefix noSimpleTemplates \
#--query_templates 1a,2a,2b,2c,6a,7a,8a

python3 main.py --db_name imdb -n -1 --algs nn --qopt_use_java 0 --test 1 \
--result_dir hyperparameter_results --eval_epoch 2 \
--sampling_priority_alpha 2.0 --eval_epoch_jerr 2 \
--num_hidden_layers 1 --max_discrete_featurizing_buckets 1 \
--hidden_layer_size 256 \
--reprioritize_epoch 2 \
--heuristic_features 1 --nn_type mscn --avg_jl_priority 1 \
--eval_test_while_training 0 --lr 0.0001 --max_epochs 40 \
--priority_err_divide_len 0 --exp_prefix noSimpleTemplates \
--query_templates 1a,2a,2b,2c,6a,7a,8a

#python3 main.py --db_name imdb -n -1 --algs nn --qopt_use_java 0 --test 1 \
#--result_dir hyperparameter_results --eval_epoch 1 \
#--sampling_priority_alpha 2.0 --eval_epoch_jerr 1 \
#--num_hidden_layers 1 --max_discrete_featurizing_buckets 1 \
#--hidden_layer_size 256 \
#--heuristic_features 1 --nn_type mscn --avg_jl_priority 0 \
#--eval_test_while_training 1 --lr 0.0001 --max_epochs 40 \
#--reprioritize_epoch 2 \
#--priority_err_divide_len 0 --exp_prefix No_Avg \
