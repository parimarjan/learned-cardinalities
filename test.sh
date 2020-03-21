python3 main.py --db_name imdb --query_dir our_dataset/queries/ -n -1 --algs nn --losses qerr,join-loss  --test 1 --max_epochs 20 --tfboard 1 --train_card_key wanderjoin10.0 --eval_test_while_training 1 --sampling_priority_alpha 0.0 --prioritize_epoch 1 --reprioritize_epoch 5 --exp_prefix wj_training --max_discrete_featurizing_buckets 1

python3 main.py --db_name imdb --query_dir our_dataset/queries/ -n -1 --algs nn --losses qerr,join-loss  --test 1 --max_epochs 20 --tfboard 1 --train_card_key wanderjoin10.0 --eval_test_while_training 1 --sampling_priority_alpha 2.0 --prioritize_epoch 1 --reprioritize_epoch 1 --exp_prefix wj_training-reprioritize --max_discrete_featurizing_buckets 1


python3 main.py --db_name imdb --query_dir our_dataset/queries/ -n -1 --algs nn --losses qerr,join-loss  --test 1 --max_epochs 20 --tfboard 1 --train_card_key wanderjoin10.0 --eval_test_while_training 1 --sampling_priority_alpha 0.0 --prioritize_epoch 1 --reprioritize_epoch 1 --exp_prefix wj_training-noreprioritize --max_discrete_featurizing_buckets 1

python3 main.py --db_name imdb --query_dir our_dataset/queries/ -n -1 --algs nn --losses qerr,join-loss  --test 1 --max_epochs 20 --tfboard 1 --train_card_key wanderjoin10.0 --eval_test_while_training 1 --sampling_priority_alpha 2.0 --prioritize_epoch 1 --reprioritize_epoch 1 --exp_prefix wj_training-buckets10 --max_discrete_featurizing_buckets 10
