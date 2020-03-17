CARD_TYPE=wanderjoin
KEY_NAME=actual
NUM_PROC=$1
PORT=5432
NUM_SAMPLES=-1
WALK_TIMEOUTS=(0.5 1.0 5.0 10.0)
TEMPLATES=(1a 2a 2b 2c 3a 4a 5a 8a)
SEEDS=(1234)
echo "num proc: " $NUM_PROC

TEMPLATES1=(1a 2c 4a)
TO=0.5
for i in "${TEMPLATES1[@]}";
  do
  echo $i;
  time python3 scripts/get_query_cardinalities.py \
  --query_dir "our_dataset/queries/${i}" \
  --card_type $CARD_TYPE --db_name imdb --port $PORT \
  --num_proc $NUM_PROC -n $NUM_SAMPLES \
  --use_tries 1 \
  --wj_walk_timeout $TO >> ./imdb_logs/$i.logs
done

TEMPLATES2=(2a 2c)
TO=1.0
for i in "${TEMPLATES2[@]}";
  do
  echo $i;
  time python3 scripts/get_query_cardinalities.py \
  --query_dir "our_dataset/queries/${i}" \
  --card_type $CARD_TYPE --db_name imdb --port $PORT \
  --num_proc $NUM_PROC -n $NUM_SAMPLES \
  --use_tries 1 \
  --wj_walk_timeout $TO >> ./imdb_logs/$i.logs
done

TEMPLATES2=(2b 3a)
TO=5.0
for i in "${TEMPLATES2[@]}";
  do
  echo $i;
  time python3 scripts/get_query_cardinalities.py \
  --query_dir "our_dataset/queries/${i}" \
  --card_type $CARD_TYPE --db_name imdb --port $PORT \
  --num_proc $NUM_PROC -n $NUM_SAMPLES \
  --use_tries 1 \
  --wj_walk_timeout $TO >> ./imdb_logs/$i.logs
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
    #--use_tries 1 \
    #--wj_walk_timeout $TO >> ./imdb_logs/$i.logs
  #done
#done
