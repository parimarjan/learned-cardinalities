#CARD_TYPE=$1
#KEY_NAME=$2
NUM_PROC=4
NUM_SAMPLES=-1

TEMPLATES=(1a 2a 2b 2c 3a 3b 4a 5a 6a 7a 8a 9a 9b 10a 11a 11b)
#TEMPLATES=(2a 2b 2c 3a 4a 5a 6a 7a 8a 9a 9b 10a 11a 11b)
#TEMPLATES=(3b)

for i in "${TEMPLATES[@]}";
  do
  echo $i;
  #python3 scripts/get_query_join_bitmaps.py \
  #--query_dir queries/imdb-unique-plans/$i/ \
  #-n $NUM_SAMPLES --no_parallel 1
  #--num_proc $NUM_PROC
  python3 scripts/save_sample_bitmaps.py \
  --query_dir queries/imdb-unique-plans/$i/ \
  -n $NUM_SAMPLES --bitmap_dir join_bitmaps_up
done

