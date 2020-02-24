python3 main.py --db_name imdb --query_dir our_dataset/queries/ -n -1 --algs nn \
--losses qerr,join-loss  --test 1 --max_epochs 40 --tfboard 1 \
--sampling_priority_type query --prioritize_epoch 1 \
--reprioritize_epoch 1 --result_dir subq_loss_cmp_results \
--sampling_priority_alpha 2.0 \
--max_discrete_featurizing_buckets 1 --exp_prefix repr1 \

python3 main.py --db_name imdb --query_dir our_dataset/queries/ -n -1 --algs nn \
--losses qerr,join-loss  --test 1 --max_epochs 40 --tfboard 1 \
--sampling_priority_type query --prioritize_epoch 1 \
--sampling_priority_alpha 2.0 \
--reprioritize_epoch 1000 --result_dir subq_loss_cmp_results \
--max_discrete_featurizing_buckets 1 --exp_prefix noRepr \

python3 main.py --db_name imdb --result_dir subq_loss_results --algs nn \
--query_dir our_dataset/queries -n -1 --sampling_priority_alpha 2.0 --losses \
qerr,join-loss --sampling_priority_type subquery --prioritize_epoch 1 \
--reprioritize_epoch 1 --max_discrete_featurizing_buckets 1 --exp_prefix subQRepr1 \
