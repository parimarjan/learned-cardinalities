python3 scripts/gen_queries.py --template_dir templates/imdb_templates/9a -n 1000 \
--query_output_dir our_dataset/queries
bash get_cards.sh our_dataset/queries/9a -1
