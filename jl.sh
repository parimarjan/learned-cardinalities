#!/bin/bash

# 1: jl_variant
# 2: max_iter
# 3: jl_start_iter
# 4: num samples
# 5: nn_cache_dir
# 6: sampling strategy: query OR subquery
# 7: test size
# 8: eval_test_while_training
# 9: lr
# 10: sampling_priority_method
# 11: sampling_priority_alpha
# 12: adaptive_priority_alpha

# TODO: change all other variables to be from command line as well, and set
# defaults

python3 main.py --db_name imdb --template_dir templates/toml2 --losses \
qerr,join-loss --qopt_java_output 0 --cache_dir ./caches -n $4 \
--use_subqueries 1 --test 1 --test_size $7 --qopt_exh 1 \
--qopt_scan_cost_factor 0.2 --results_cache jl_results \
--qopt_get_sql 0 --algs nn2 --jl_variant $1 --jl_start_iter $3 \
--max_iter $2 --eval_iter 1000 --loss_func qloss --sampling $6 \
--nn_cache_dir $5 --eval_test_while_training $8 --lr $9 \
--sampling_priority_method ${10} --sampling_priority_alpha ${11} \
--adaptive_priority_alpha ${12}
