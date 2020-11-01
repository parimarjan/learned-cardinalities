LOSS=$1
NN_TYPE=$2
LR=$3
HLS=64
#DECAY=1.0
BUCKETS=10

#RES_DIR=all_results/vldb/so/test_diff/pr/final_mscn1
RES_DIR=all_results/vldb/so/test_diff3/mscn/final1
#RES_DIR=all_results/vldb/so/test_diff2/fcnn/final1

DECAY=1.0

bash so_run_all_diff.sh $LOSS $LR 0.50 $HLS $BUCKETS 10 $NN_TYPE \
  $RES_DIR $DECAY

bash so_run_all_diff.sh $LOSS $LR 0.50 $HLS $BUCKETS 10 $NN_TYPE \
  $RES_DIR 0.1

bash so_run_all_diff.sh $LOSS $LR 0.50 $HLS $BUCKETS 20 $NN_TYPE \
  $RES_DIR $DECAY

bash so_run_all_diff.sh $LOSS $LR 0.50 $HLS $BUCKETS 20 $NN_TYPE \
  $RES_DIR $DECAY

#bash so_run_all_diff.sh $LOSS $LR 0.50 $HLS $BUCKETS 30 $NN_TYPE \
  #$RES_DIR $DECAY

