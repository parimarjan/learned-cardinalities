#CARD_TYPE=$1
#KEY_NAME=$2
NUM_PROC=-1
#queries/synth_mix4
#echo "card_type: $CARD_TYPE, key_name: $KEY_NAME, num_processes: $NUM_PROC , num_samples:  $NUM_SAMPLES"
#python3 scripts/get_query_cardinalities.py --query_dir
#../MyCEB/queries/synth_mix3/6 --card_type pg --key_name expected --db_name
#synth1


#TEMPLATES=(1 2 3 4 5 6 7 8 9 10 2stdoutlier)
#TEMPLATES=(1 2)
#TEMPLATES=(4b 5 6 7 7b 7c)
#TEMPLATES=(4b 5 6 7 7c)
#TEMPLATES=(7 7c)

#DB=synth1
#DB=ergastf1

DB=imdb
TEMPLATES=(1a 2b 2c 3a 4a 5a 7a 8a 9a 9b 10a 10b 11a 11b)
NUM_SAMPLES=-1
for i in "${TEMPLATES[@]}";
  do
  echo $i;
  python3 scripts/get_query_cardinalities.py \
  --query_dir ../MyCEB/queries/mssql/imdb-unique-plans/$i \
  --card_type pg --db_name $DB --port 5432 -n $NUM_SAMPLES \
  --key_name expected --true_timeout 900000 \
  --num_proc $NUM_PROC

  python3 scripts/get_query_cardinalities.py \
  --query_dir ../MyCEB/queries/mssql/imdb-unique-plans/$i \
  --skip_zero_queries 0 \
  --card_type actual --db_name $DB --port 5432 -n $NUM_SAMPLES \
  --key_name actual --true_timeout 900000 \
  --num_proc $NUM_PROC
done


