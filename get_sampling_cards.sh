CARD_TYPE=wanderjoin
KEY_NAME=actual
NUM_PROC=$1
NUM_SAMPLES=-1
WALK_TIMEOUTS=(10.0 1.0 20.0)
TEMPLATES=(1a 2a 2b 2c 3a 4a 5a 6a 8a)
echo "num proc: " $NUM_PROC

for TO in "${WALK_TIMEOUTS[@]}";
  do
  echo "walk timeout: " $TO
  for i in "${TEMPLATES[@]}";
    do
    echo $i;
    python3 scripts/get_query_cardinalities.py \
    --query_dir "our_dataset/queries/${i}" \
    --card_type $CARD_TYPE --db_name imdb --port 5432 \
    --num_proc $NUM_PROC -n $NUM_SAMPLES \
    --wj_walk_timeout $TO >> ./imdb_logs/$i.logs
  done
done
