
CARD_TYPE=$1
KEY_NAME=$2
NUM_PROC=$3
NUM_SAMPLES=$4
echo "card_type: $CARD_TYPE, key_name: $KEY_NAME, num_processes: $NUM_PROC , num_samples:  $NUM_SAMPLES"

for i in {1..18};
	do
	echo $i;
  python3 scripts/get_query_cardinalities.py --query_dir so_workload/$i \
  --card_type $CARD_TYPE --db_name so --port 5432 -n $NUM_SAMPLES \
  --key_name $KEY_NAME --true_timeout 900000 \
  --num_proc $NUM_PROC >> ./logs/$i.logs
done

#for i in {1..18};
	#do
	#echo $i;
  #python3 scripts/get_query_cardinalities.py --query_dir so_workload/$i \
  #--card_type $CARD_TYPE --db_name so --port 5433 -n 1000 \
  #--key_name $KEY_NAME --true_timeout 900000 \
  #--num_proc $NUM_PROC >> ./logs/$i.logs
#done

