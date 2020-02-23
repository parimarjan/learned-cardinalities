
#python3 main.py --db_name imdb --result_dir subq_loss_results --algs nn \
#--query_dir our_dataset/queries -n -1 --sampling_priority_alpha 2.0 --losses \
#qerr,join-loss --sampling_priority_type query --prioritize_epoch 1 \
#--reprioritize_epoch 1 --max_discrete_featurizing_buckets 1 --exp_prefix repr1 \

#python3 main.py --db_name imdb --result_dir subq_loss_results --algs nn \
#--query_dir our_dataset/queries -n -1 --sampling_priority_alpha 2.0 --losses \
#qerr,join-loss --sampling_priority_type query --prioritize_epoch 1 \
#--reprioritize_epoch 1000 --max_discrete_featurizing_buckets 1 --exp_prefix noRepr \

python3 main.py --db_name imdb --result_dir subq_loss_results --algs nn \
--query_dir our_dataset/queries -n -1 --sampling_priority_alpha 2.0 --losses \
qerr,join-loss --sampling_priority_type subquery --prioritize_epoch 1 \
--reprioritize_epoch 1 --max_discrete_featurizing_buckets 1 --exp_prefix subQRepr1 \

#python3 main.py --db_name imdb -n -1 --algs nn --qopt_use_java 0 --test 1 \
#--result_dir plan_viz_results --eval_epoch 2 \
#--sampling_priority_alpha 2.0 --eval_epoch_jerr 2 \
#--num_hidden_layers 1 --max_discrete_featurizing_buckets 1 \
#--hidden_layer_size 128 \
#--reprioritize_epoch 2 \
#--heuristic_features 1 --nn_type mscn --avg_jl_priority 1 \
#--eval_test_while_training 1 --lr 0.0001 --max_epochs 40 \
#--priority_err_divide_len 0 --exp_prefix planViz \
#--prioritize_epoch 2

#python3 main.py --db_name imdb -n -1 --algs nn --qopt_use_java 0 --test 1 \
#--result_dir plan_viz_results --eval_epoch 2 \
#--sampling_priority_alpha 0.0 --eval_epoch_jerr 2 \
#--num_hidden_layers 1 --max_discrete_featurizing_buckets 1 \
#--hidden_layer_size 128 \
#--reprioritize_epoch 2 \
#--heuristic_features 1 --nn_type mscn --avg_jl_priority 1 \
#--eval_test_while_training 1 --lr 0.0001 --max_epochs 40 \
#--priority_err_divide_len 0 --exp_prefix planViz \
#--prioritize_epoch 2

# subquery priority example
#python3 main.py --db_name imdb -n -1 --algs nn --qopt_use_java 0 --test 1 \
#--result_dir plan_comparison_results --eval_epoch 2 \
#--sampling_priority_alpha 2.0 --eval_epoch_jerr 2 \
#--num_hidden_layers 1 --max_discrete_featurizing_buckets 1 \
#--hidden_layer_size 32 \
#--reprioritize_epoch 2 \
#--heuristic_features 1 --nn_type mscn --avg_jl_priority 1 \
#--eval_test_while_training 1 --lr 0.0001 --max_epochs 40 \
#--priority_err_divide_len 0 --exp_prefix subQuerySamplingPriority \
#--prioritize_epoch 1 --reprioritize_epoch 1000 \
#--sampling_priority_type subquery

