BUCKET=$1
PRIORITY=$2
JLPN=$3
NUM_GROUPS=1
NUM_BUCKETS=(1 10 25 50)
HLS=(64 128 256 512)
REPR=1
MAX_EPOCHS=20
RESULT_DIR=debug_results
DEBUG_SET=0
PRS=(0.25 0.75 1.0 2.0)
#OPTS=(adam adamw ams)
OPTS=(ams adamw)
HLS_FIXED=128

for i in "${!PRS[@]}";
  do
  for j in "${!OPTS[@]}";
    do
      time python3 main.py --algs nn --sampling_priority_alpha ${PRS[$i]} \
      --max_discrete_featurizing_buckets $BUCKET \
      --hidden_layer_size $HLS_FIXED --max_epochs $MAX_EPOCHS \
      --reprioritize_epoch $REPR \
      --prioritize_epoch $REPR \
      --exp_prefix no7a-${OPTS[$j]}-${PRS[$i]} \
      --join_loss_pool_num $JLPN \
      --test 1 --eval_epoch_jerr 200 \
      --eval_epoch 200 \
      --num_groups $NUM_GROUPS \
      --loss_func mse \
      --normalization_type mscn \
      --result_dir $RESULT_DIR \
      --debug_set $DEBUG_SET \
      --optimizer_name ${OPTS[$j]}
  done
done
