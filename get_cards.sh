
CARD_TYPE=actual
KEY_NAME=actual
NUM_PROC=$1
echo $NUM_PROC
for i in {14..18};
	do
	echo $i;
  python3 scripts/get_query_cardinalities.py --query_dir so_workload/$i \
  --card_type $CARD_TYPE --db_name so --port 5433 -n 500 \
  --key_name $KEY_NAME --true_timeout 900000 \
  --num_proc $NUM_PROC >> ./logs/$i.logs
done

for i in {13..18};
	do
	echo $i;
  python3 scripts/get_query_cardinalities.py --query_dir so_workload/$i \
  --card_type $CARD_TYPE --db_name so --port 5433 -n 1000 \
  --key_name $KEY_NAME --true_timeout 900000 \
  --num_proc $NUM_PROC >> ./logs/$i.logs
done

