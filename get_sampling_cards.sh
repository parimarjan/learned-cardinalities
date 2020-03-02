TIMEOUT=180000
SP=25
CARD_TYPE=actual
KEY_NAME=actual
NUM_PROC=$1
NUM_SAMPLES=500
TEMPLATES=(1a 2a 2b 2c 3a 4a 5a 6a 8a)
#TEMPLATES=(4a 5a 6a 8a)
echo "num proc: " $NUM_PROC
for i in "${TEMPLATES[@]}";
	do
	echo $i;
  python3 scripts/get_query_cardinalities.py \
  --query_dir "our_dataset/queries/${i}" \
  --card_type $CARD_TYPE --db_name imdb --port 5433 -n -1 \
  --key_name $KEY_NAME --true_timeout $TIMEOUT \
  --num_proc $NUM_PROC --sampling_type ss -n $NUM_SAMPLES \
  --sampling_percentage $SP >> ./imdb_logs/$i.logs
done

#python3 scripts/get_query_cardinalities.py --query_dir our_dataset/queries/1a \
#--sampling_percentage 25 --sampling_type ss --card_type actual --port 5433 \
#--num_proc -1 --true_timeout $TIMEOUT

