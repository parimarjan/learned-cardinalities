#for i in {0..1}; do python3 main.py --db_name imdb --template_dir \
  #templates/all_toml7/ --losses qerr,join-loss --qopt_java_output 0 \
#--cache_dir /data/pari/caches -n -1 --use_subqueries 1 --test 1 --test_size 0.5 \
#--qopt_scan_cost_factor 0.2 --results_cache debug --qopt_get_sql 0 \
#--algs nn2 --eval_iter 1000 --eval_iter_jl 5000 --max_iter 250000 \
#--reuse_env 0 --lr 0.001 --eval_num_tables 1; done

for i in {0..0}; do python3 main.py --db_name imdb --template_dir \
  templates/all_tomls/ --losses qerr,join-loss --qopt_java_output 0 \
--cache_dir /data/pari/caches -n -1 --use_subqueries 1 --test 1 --test_size 0.5 \
--qopt_scan_cost_factor 0.2 --results_cache debug --qopt_get_sql 0 \
--algs nn2 --eval_iter 1000 --eval_iter_jl 10000 --max_iter 250000 \
--reuse_env 0 --lr 0.001 --eval_num_tables 1; done
