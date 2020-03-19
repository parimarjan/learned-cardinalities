
python3 main.py --algs nn --test 1 --max_epochs 20 --train_card_key actual \
--eval_test_while_training 1 --sampling_priority_alpha 0.0 \
--exp_prefix baseline

python3 main.py --algs nn --test 1 --max_epochs 20 --train_card_key actual \
--eval_test_while_training 1 --sampling_priority_alpha 2.0 \
--exp_prefix baselinePriority

python3 main.py --algs nn --test 1 --max_epochs 20 --train_card_key actual \
--eval_test_while_training 1 --sampling_priority_alpha 2.0 \
--exp_prefix divideLenPriority --priority_err_divide_len 1

python3 main.py --algs nn --test 1 --max_epochs 20 --train_card_key wanderjoin \
--eval_test_while_training 1 --sampling_priority_alpha 2.0 \
--exp_prefix wj

python3 main.py --algs nn --test 1 --max_epochs 20 --train_card_key wanderjoin0.5 \
--eval_test_while_training 1 --sampling_priority_alpha 2.0 \
--exp_prefix wj0.5

