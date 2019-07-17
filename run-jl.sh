CUDA_VISIBLE_DEVICES=2 time python3 main.py --db_name imdb --template_dir templates/myjob -n 1 \
--algs nn2 --test 0 --use_subqueries 1 --only_nonzero_samples 0 \
--losses qerr,join-loss --qopt_java_output 0 --qopt_left_deep 0 \
--qopt_reward_normalization "" --qopt_exh 1 --qopt_recompute_fixed_planners 1 \
--baseline_join_alg "EXHAUSTIVE" \
--max_iter 20000 --jl_variant 2 --qopt_verbose 1 \
--qopt_log_file "./java-jl2.log" --eval_iter 500 \
--num_hidden_layer 1 --optimizer_name adam --jl_start_iter 400 \
--cache_dir /data/pari/dbs/caches/ --qopt_log_file "java-jl.log" \
--divide_mb_len 1

CUDA_VISIBLE_DEVICES=2 time python3 main.py --db_name imdb --template_dir templates/myjob -n 1 \
--algs nn2 --test 0 --use_subqueries 1 --only_nonzero_samples 0 \
--losses qerr,join-loss --qopt_java_output 0 --qopt_left_deep 0 \
--qopt_reward_normalization "" --qopt_exh 1 --qopt_recompute_fixed_planners 1 \
--baseline_join_alg "EXHAUSTIVE" \
--max_iter 20000 --jl_variant 2 --qopt_verbose 1 \
--qopt_log_file "./java-jl2.log" --eval_iter 500 \
--num_hidden_layer 1 --optimizer_name adam --jl_start_iter 400 \
--cache_dir /data/pari/dbs/caches/ \
--divide_mb_len 1

CUDA_VISIBLE_DEVICES=2 time python3 main.py --db_name imdb --template_dir templates/myjob -n 1 \
--algs nn2 --test 0 --use_subqueries 1 --only_nonzero_samples 0 \
--losses qerr,join-loss --qopt_java_output 0 --qopt_left_deep 0 \
--qopt_reward_normalization "" --qopt_exh 1 --qopt_recompute_fixed_planners 1 \
--baseline_join_alg "EXHAUSTIVE" \
--max_iter 20000 --jl_variant 2 --qopt_verbose 1 \
--qopt_log_file "./java-jl2.log" --eval_iter 500 \
--num_hidden_layer 1 --optimizer_name adam --jl_start_iter 400 \
--cache_dir /data/pari/dbs/caches/ \
--divide_mb_len 1
