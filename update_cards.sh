
TEMPLATES=(1 2 11 12 13 14 15 16 17 18)

for tmp in "${TEMPLATES[@]}";
do
  python3 scripts/update_qreps.py --query_dir so_workload/${tmp}
  #python3 scripts/update_qreps.py --query_dir our_dataset/queries/${tmp}
done
