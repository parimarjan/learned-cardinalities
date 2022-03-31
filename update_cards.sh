
#TEMPLATES=(1 2 11 12 13 14 15 16 17 18)
#TEMPLATES=(2b 3a 4a 5a 6a 7a 8a 9a 10a 10b 11a)
TEMPLATES=(1a 2a 10a 11a 11b 2b 2c 3a 3b 4a 5a 6a 7a 8a 9a 9b)

for tmp in "${TEMPLATES[@]}";
do
  #python3 scripts/update_qreps.py --query_dir so_workload/${tmp}
  #python3 scripts/update_qreps.py --query_dir our_dataset/queries/${tmp}
  #rm -rf our_dataset/queries/${tmp}/wj_data

  python3 scripts/save_sample_bitmaps.py \
  --query_dir queries/imdb-unique-plans/${tmp} \
  --bitmap_dir sample_bitmaps_uniquep

  #python3 scripts/get_query_sample_bitmaps.py -n -1 \
  #--query_dir queries/imdb-unique-plans/${tmp} \
  #--num_proc 1 --no_par 1
done
