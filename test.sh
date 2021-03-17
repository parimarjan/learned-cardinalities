python3 main.py --algs true --query_template all -n -1 --debug_set 1 \
--eval_epoch 100 --losses mysql-loss,qerr --query_dir queries/imdb \
--result_dir debug_results

python3 main.py --algs postgres --query_template all -n -1 --debug_set 1 \
--eval_epoch 100 --losses mysql-loss,qerr --query_dir queries/imdb \
--result_dir debug_results
