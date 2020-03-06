CARD_TYPE=wanderjoin
KEY_NAME=actual
NUM_PROC=$1
NUM_SAMPLES=$2
WALK_TIMEOUTS=(10.0)
#TEMPLATES=(1a 6a 8a)
TEMPLATES=(2b)
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
      --card_type $CARD_TYPE --db_name imdb --port 5433 \
      --num_proc $NUM_PROC -n $NUM_SAMPLES \
      --seed $SEED \
      --wj_walk_timeout $TO >> ./imdb_logs/$i.logs
    done
  done
done

#for TO in "${WALK_TIMEOUTS2[@]}";
  #do
  #echo "walk timeout: " $TO
  #for i in "${TEMPLATES2[@]}";
    #do
    #echo $i;
    #python3 scripts/get_query_cardinalities.py \
    #--query_dir "our_dataset/queries/${i}" \
    #--card_type $CARD_TYPE --db_name imdb --port 5432 \
    #--num_proc $NUM_PROC -n $NUM_SAMPLES \
    #--wj_walk_timeout $TO >> ./imdb_logs/$i.logs
  #done
#done
