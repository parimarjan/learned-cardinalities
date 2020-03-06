CARD_TYPE=wanderjoin
KEY_NAME=actual
NUM_PROC=$1
PORT=5432
NUM_SAMPLES=200
NUM_SAMPLES2=-1
WALK_TIMEOUTS=(0.5 1.0 5.0 10.0 25.0)
TEMPLATES=(1a 2a 2b 2c 3a 4a 6a 8a)
SEEDS=(1234)
echo "num proc: " $NUM_PROC

for SEED in "${SEEDS[@]}";
  do
  echo "seed: " $SEED
  for TO in "${WALK_TIMEOUTS[@]}";
    do
    echo "walk timeout: " $TO
    for i in "${TEMPLATES[@]}";
      do
      echo $i;
      time python3 scripts/get_query_cardinalities.py \
      --query_dir "our_dataset/queries/${i}" \
      --card_type $CARD_TYPE --db_name imdb --port $PORT \
      --num_proc $NUM_PROC -n $NUM_SAMPLES \
      --seed $SEED \
      --wj_walk_timeout $TO >> ./imdb_logs/$i.logs
    done
  done
done

#for TO in "${WALK_TIMEOUTS[@]}";
  #do
  #echo "walk timeout: " $TO
  #for i in "${TEMPLATES[@]}";
    #do
    #echo $i;
    #python3 scripts/get_query_cardinalities.py \
    #--query_dir "our_dataset/queries/${i}" \
    #--card_type $CARD_TYPE --db_name imdb --port $PORT \
    #--num_proc $NUM_PROC -n $NUM_SAMPLES2 \
    #--wj_walk_timeout $TO >> ./imdb_logs/$i.logs
  #done
#done
