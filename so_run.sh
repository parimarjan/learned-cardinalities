LOSS=$1
NN_TYPE=$2
LR=0.0001
DECAY=1.0
HLS=128
BUCKETS=10
EPOCHS=50

RES_DIR=all_results/vldb/so/test_diff/mscn/run3
RES_DIR2=all_results/vldb/so/test_diff/mscn/run2

#DECAY=1.0
#DECAY2=0.1

bash so_run_all_diff.sh $LOSS $LR 0.50 $HLS $BUCKETS $EPOCHS $NN_TYPE \
  $RES_DIR $DECAY

bash so_run_all_diff.sh $LOSS $LR 0.50 $HLS $BUCKETS $EPOCHS $NN_TYPE \
  $RES_DIR $DECAY

bash so_run_all_diff.sh $LOSS $LR 0.30 $HLS $BUCKETS $EPOCHS $NN_TYPE \
  $RES_DIR2 $DECAY

bash so_run_all_diff.sh $LOSS $LR 0.30 $HLS $BUCKETS $EPOCHS $NN_TYPE \
  $RES_DIR2 $DECAY

bash so_run_all_diff.sh $LOSS $LR 0.50 $HLS $BUCKETS 40 $NN_TYPE \
  $RES_DIR $DECAY

bash so_run_all_diff.sh $LOSS $LR 0.50 $HLS $BUCKETS 40 $NN_TYPE \
  $RES_DIR $DECAY

