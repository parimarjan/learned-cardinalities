#CARD_TYPE=$1
#KEY_NAME=$2
#NUM_PROC=$3
#NUM_SAMPLES=$4
#echo "card_type: $CARD_TYPE, key_name: $KEY_NAME, num_processes: $NUM_PROC , num_samples:  $NUM_SAMPLES"

#TEMPLATES=(1 11 12 13 15 16 17 18)
#for i in "${TEMPLATES[@]}";
	#do
	#echo $i;
  #python3 scripts/get_query_cardinalities.py --query_dir so_workload/$i \
  #--card_type $CARD_TYPE --db_name so --port 5432 -n $NUM_SAMPLES \
  #--key_name $KEY_NAME --true_timeout 900000 \
  #--num_proc $NUM_PROC >> ./logs/$i.logs
#done

QUERY_DIR=$1
NUM_SAMPLES=$2
python3 scripts/get_query_cardinalities.py --query_dir $QUERY_DIR \
--card_type total -n $NUM_SAMPLES

python3 scripts/get_query_cardinalities.py --query_dir $QUERY_DIR \
--card_type pg --key_name expected -n $NUM_SAMPLES

python3 scripts/get_query_cardinalities.py --query_dir $QUERY_DIR \
--card_type actual -n $NUM_SAMPLES


