BUCKET=$1
PRIORITY=$2
JLPN=$3
NUM_GROUPS=1
#NUM_BUCKETS=(1 10 25 50)
#NUM_BUCKETS=(25 50 10 1)
#NUM_BUCKETS=(25 50 10 100)
#NUM_BUCKETS=(25 50 10)
NUM_BUCKETS=(100)
HLS=(128 256 512)
HLS=(1024)
REPR=1
MAX_EPOCHS=15

HLS_FIXED=256
echo "going to run all configurations w/o priority"
for j in "${!NUM_BUCKETS[@]}";
  do
  for i in "${!HLS[@]}";
    do
    echo $i $j ", hls:" "${HLS[$i]}" "buckets:" "${NUM_BUCKETS[$j]}"
    time python3 main.py --algs nn --sampling_priority_alpha $PRIORITY \
    --max_discrete_featurizing_buckets ${NUM_BUCKETS[$j]} \
    --hidden_layer_size ${HLS[$i]} --max_epochs $MAX_EPOCHS \
    --reprioritize_epoch $REPR \
    --prioritize_epoch 2 \
    --exp_prefix allRuns \
    --join_loss_pool_num $JLPN \
    --tfboard 0 --test 1 --eval_epoch_jerr 200 \
    --eval_epoch 200 \
    --num_groups $NUM_GROUPS \
    --loss_func mse \
    --normalization_type mscn \
    --eval_test_while_training 0 \
    --result_dir new_results2 \
    --weight_decay 0.0 \
    --use_val_set 1
  done
done
