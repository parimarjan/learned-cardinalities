python3 main.py --db_name imdb --query_dir our_dataset/queries --algs true \
--losses qerr,join-loss --sampling_priority_alpha 2.0 --max_epochs 10 \
--query_templates 1a

python3 main.py --db_name imdb --query_dir our_dataset/queries --algs postgres \
--losses qerr,join-loss --sampling_priority_alpha 2.0 --max_epochs 10 \
--query_templates 1a

#python3 main.py --db_name imdb --query_dir our_dataset/queries --algs nn \
#--losses qerr,join-loss --sampling_priority_alpha 2.0 --max_epochs 10 \
#--query_templates 1a

#python3 main.py --db_name imdb --query_dir our_dataset/queries --algs nn \
#--losses qerr,join-loss --sampling_priority_alpha 10.0 --max_epochs 10 \
#--query_templates 1a

#python3 main.py --db_name imdb --query_dir our_dataset/queries --algs nn \
#--losses qerr,join-loss --sampling_priority_alpha 4.0 --max_epochs 10 \
#--query_templates 1a

#python3 main.py --db_name imdb --query_dir our_dataset/queries --algs nn \
#--losses qerr,join-loss --sampling_priority_alpha 3.0 --max_epochs 10 \
#--query_templates 1a

#python3 main.py --db_name imdb --query_dir our_dataset/queries --algs nn \
#--losses qerr,join-loss --sampling_priority_alpha 6.0 --max_epochs 10 \
#--query_templates 1a

