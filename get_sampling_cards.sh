CARD_TYPE=wanderjoin
#KEY_NAME=actual
NUM_PROC=$1
PORT=5432
NUM_SAMPLES=200
TEMPLATES=(1a 2a 2b 2c 3a 4a 5a 6a 8a)
#TOS=(0.25 0.5 1.0 0.5 0.25 0.1 0.1 1.0 5.0)
TOS=(0.12 0.25 0.5 0.25 0.12 0.05 0.05 0.5 2.5)

SEEDS=(1234)
echo "num proc: " $NUM_PROC

#for i in "${!TEMPLATES[@]}";
  #do
  #echo $i "${TEMPLATES[$i]}" "${TOS[$i]}"
  #time python3 scripts/get_query_cardinalities.py \
  #--query_dir "our_dataset/queries/${TEMPLATES[$i]}" \
  #--card_type $CARD_TYPE --db_name imdb --port $PORT \
  #--num_proc $NUM_PROC -n $NUM_SAMPLES \
  #--use_tries 1 \
  #--wj_walk_timeout ${TOS[$i]} >> ./imdb_logs/$i.logs
#done

for i in "${!TEMPLATES[@]}";
  do
  echo $i "${TEMPLATES[$i]}" "${TOS[$i]}"
  time python3 scripts/get_query_cardinalities.py \
  --query_dir "our_dataset/queries/${TEMPLATES[$i]}" \
  --card_type $CARD_TYPE --db_name imdb --port $PORT \
  --num_proc $NUM_PROC -n -1 \
  --use_tries 1 \
  --wj_walk_timeout ${TOS[$i]} >> ./imdb_logs/$i.logs
done

