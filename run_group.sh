#python3 main.py --db_name imdb --template_dir templates/all_tomls/ --losses qerr --qopt_java_output 0 --cache_dir caches -n -1 --use_subqueries 1 --test 1 --test_size 0.5 --qopt_scan_cost_factor 0.2 --results_cache debug --qopt_get_sql 0 --algs nn3 --net_name FCNN --eval_iter 1000 --eval_iter_jl 5000 --lr 0.001 --group_models -2 --eval_num_tables 1 --max_iter 50000

python3 main.py --db_name imdb --template_dir templates/all_tomls/ --losses \
qerr --qopt_java_output 0 --cache_dir caches -n -1 --use_subqueries 1 \
--test 1 --test_size 0.5 --qopt_scan_cost_factor 0.2 --results_cache debug \
--qopt_get_sql 0 --algs nn3 --net_name FCNN --eval_iter 1000 --eval_iter_jl \
5000 --lr 0.001 --group_models -4 --eval_num_tables 1 --max_iter 25100 \

python3 main.py --db_name imdb --template_dir templates/all_tomls/ --losses \
qerr --qopt_java_output 0 --cache_dir caches -n -1 --use_subqueries 1 \
--test 1 --test_size 0.5 --qopt_scan_cost_factor 0.2 --results_cache debug \
--qopt_get_sql 0 --algs nn3 --net_name FCNN --eval_iter 1000 --eval_iter_jl \
5000 --lr 0.001 --group_models -6 --eval_num_tables 1 --max_iter 25100 \

#python3 main.py --db_name imdb --template_dir templates/all_tomls/ --losses \
#qerr --qopt_java_output 0 --cache_dir caches -n -1 --use_subqueries 1 \
#--test 1 --test_size 0.5 --qopt_scan_cost_factor 0.2 --results_cache debug \
#--qopt_get_sql 0 --algs nn3 --net_name FCNN --eval_iter 1000 --eval_iter_jl \
#5000 --lr 0.001 --group_models -8 --eval_num_tables 1 --max_iter 25100 \

#python3 main.py --db_name imdb --template_dir templates/all_tomls/ --losses \
#qerr --qopt_java_output 0 --cache_dir caches -n -1 --use_subqueries 1 \
#--test 1 --test_size 0.5 --qopt_scan_cost_factor 0.2 --results_cache debug \
#--qopt_get_sql 0 --algs nn3 --net_name FCNN --eval_iter 1000 --eval_iter_jl \
#5000 --lr 0.001 --group_models -10 --eval_num_tables 1 --max_iter 15100 \

#python3 main.py --db_name imdb --template_dir templates/all_tomls/ --losses \
#qerr --qopt_java_output 0 --cache_dir caches -n -1 --use_subqueries 1 \
#--test 1 --test_size 0.5 --qopt_scan_cost_factor 0.2 --results_cache debug \
#--qopt_get_sql 0 --algs nn3 --net_name FCNN --eval_iter 1000 --eval_iter_jl \
#5000 --lr 0.001 --group_models -12 --eval_num_tables 1 --max_iter 15100 \

#python3 main.py --db_name imdb --template_dir templates/all_tomls/ --losses \
#qerr --qopt_java_output 0 --cache_dir caches -n -1 --use_subqueries 1 \
#--test 1 --test_size 0.5 --qopt_scan_cost_factor 0.2 --results_cache debug \
#--qopt_get_sql 0 --algs nn3 --net_name FCNN --eval_iter 1000 --eval_iter_jl \
#5000 --lr 0.001 --group_models -14 --eval_num_tables 1 --max_iter 15100 \
