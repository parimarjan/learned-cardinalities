DIFF_SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 16 17 18 19 20 21 22 23 24 25 26
  27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50)

for k in "${!DIFF_SEEDS[@]}";
  do
  CMD="time python3 main.py --algs nn -n 2 \
   --only_compute_overlap 1 \
   --result_dir all_results/ \
   --eval_epoch 1 \
   --test_diff_templates 1 --diff_templates_type 3 \
   --diff_templates_seed ${DIFF_SEEDS[$k]}"
  echo $CMD
  eval $CMD
done
