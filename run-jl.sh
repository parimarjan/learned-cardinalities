time python3 main.py --db_name imdb --template_dir templates/myjob -n 1 \
--algs nn2 --test 0 --use_subqueries 1 --only_nonzero_samples 0 \
--losses qerr,join-loss --qopt_java_output 0 --qopt_left_deep 1 \
--qopt_reward_normalization "" --qopt_exh 0 --qopt_recompute_fixed_planners 1 \
--baseline_join_alg "LEFT_DEEP" \
--max_iter 10000 --jl_variant 3 --qopt_verbose 1 \
--qopt_log_file "./java-jl.log" --optimizer_name ams --eval_iter 500 \
--num_hidden_layer 1 --optimizer_name ams \

time python3 main.py --db_name imdb --template_dir templates/myjob -n 1 \
--algs nn2 --test 0 --use_subqueries 1 --only_nonzero_samples 0 \
--losses qerr,join-loss --qopt_java_output 0 --qopt_left_deep 1 \
--qopt_reward_normalization "" --qopt_exh 0 --qopt_recompute_fixed_planners 1 \
--baseline_join_alg "LEFT_DEEP" \
--max_iter 10000 --jl_variant 3 --qopt_verbose 1 \
--qopt_log_file "./java-jl.log" --optimizer_name ams --eval_iter 500 \
--num_hidden_layer 1 --optimizer_name ams \

time python3 main.py --db_name imdb --template_dir templates/myjob -n 1 \
--algs nn2 --test 0 --use_subqueries 1 --only_nonzero_samples 0 \
--losses qerr,join-loss --qopt_java_output 0 --qopt_left_deep 1 \
--qopt_reward_normalization "" --qopt_exh 0 --qopt_recompute_fixed_planners 1 \
--baseline_join_alg "LEFT_DEEP" \
--max_iter 10000 --jl_variant 3 --qopt_verbose 1 \
--qopt_log_file "./java-jl.log" --optimizer_name ams --eval_iter 500 \
--num_hidden_layer 1 --optimizer_name ams \
