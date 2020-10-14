#CARD_TYPE=$1
#KEY_NAME=$2
NUM_PROC=40
NUM_SAMPLES=-1

TEMPLATES=(2a 3a 4a 5a 6a 7a 8a 9a 9b 10a 11a 11b)
for i in "${TEMPLATES[@]}";
  do
  echo $i;
  python3 scripts/get_query_sample_bitmaps.py \
  --query_dir our_dataset/queries/$i/ \
  -n $NUM_SAMPLES \
  --num_proc $NUM_PROC
done

