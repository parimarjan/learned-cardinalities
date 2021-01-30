#CARD_TYPE=$1
#KEY_NAME=$2
CARD_TYPE=$1
KEY_NAME=$2
NUM_PROC=$3
#DB_YEAR=$4
NUM_SAMPLES=-1
echo "card_type: $CARD_TYPE, key_name: $KEY_NAME, num_processes: $NUM_PROC , num_samples:  $NUM_SAMPLES"

#TEMPLATES=(1a 2a 2b 2c 3a 4a 5a 9a 9b 10a 11a 11b)
TEMPLATES=(1a 2a 2b 2c)
DBYEARS=(1950 1960 1970 1980 1990 2000)

for i in "${TEMPLATES[@]}";
  do
  echo $i;
  for db in "${DBYEARS[@]}";
    do
    echo $db;
    time python3 scripts/get_query_cardinalities.py --query_dir minified_dataset/$i \
    --card_type $CARD_TYPE --db_name imdb --port 5432 -n $NUM_SAMPLES \
    --key_name $KEY_NAME --true_timeout 900000 --db_year $db \
    --num_proc $NUM_PROC >> ./logs/$i.logs
  done
done

#QUERY_DIR=$1
#NUM_SAMPLES=$2
#python3 scripts/get_query_cardinalities.py --query_dir $QUERY_DIR \
#--card_type total -n $NUM_SAMPLES

#python3 scripts/get_query_cardinalities.py --query_dir $QUERY_DIR \
#--card_type pg --key_name expected -n $NUM_SAMPLES

#python3 scripts/get_query_cardinalities.py --query_dir $QUERY_DIR \
#--card_type actual -n $NUM_SAMPLES


