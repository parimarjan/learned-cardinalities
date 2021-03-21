time python3 main.py \
--query_template 1a,2a \
-n -1 --losses mysql-cost-model --user pari --pwd password --algs true

time python3 main.py \
--query_template 2b,2c,3a,4a \
-n -1 --losses mysql-cost-model --user pari --pwd password --algs true

time python3 main.py \
--query_template 8a,9a,9b,10a,11a,11b \
-n -1 --losses mysql-cost-model --user pari --pwd password --algs true

time python3 main.py \
--query_template 5a,6a,7a \
-n -1 --losses mysql-cost-model --user pari --pwd password --algs true

