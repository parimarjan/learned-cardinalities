
#DBYEARS=(1950 1960 1970 1980 1990 2000)
DBYEARS=(1960 1970 1980)

for db in "${DBYEARS[@]}";
  do
  echo $db;
  time python3 scripts/create_dynamic_db.py --user ubuntu --create_db 1 \
  --max_year $db
done

