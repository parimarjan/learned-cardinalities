TIMEOUT=180000
SP=50
CARD_TYPE=actual
KEY_NAME=actual
NUM_PROC=$1
NUM_SAMPLES=100
echo "num proc: " $NUM_PROC
for i in {2..8};
	do
	echo $i;
  python3 scripts/get_query_cardinalities.py \
  --query_dir "our_dataset/queries/${i}a" \
  --card_type $CARD_TYPE --db_name imdb --port 5433 -n -1 \
  --key_name $KEY_NAME --true_timeout $TIMEOUT \
  --num_proc $NUM_PROC --sampling_type ss -n $NUM_SAMPLES \
  --sampling_percentage $SP >> ./imdb_logs/$i.logs
done

#python3 scripts/get_query_cardinalities.py --query_dir our_dataset/queries/1a \
#--sampling_percentage 25 --sampling_type ss --card_type actual --port 5433 \
#--num_proc -1 --true_timeout $TIMEOUT

