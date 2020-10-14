BUCKET=$1
PRIORITY=$2
JLPN=$3
MAX_EPOCHS=20
NUM_GROUPS=1
#NUM_BUCKETS=(1 10 25 50)
HLS=(64 128 256 512)
#HLS=(256)
REPR=1

for i in "${!HLS[@]}";
  do
    echo $i "${HLS[$i]}"
    time python3 main.py --algs nn --sampling_priority_alpha $PRIORITY \
    --max_discrete_featurizing_buckets $BUCKET \
    --hidden_layer_size ${HLS[$i]} --max_epochs $MAX_EPOCHS \
    --reprioritize_epoch $REPR \
    --prioritize_epoch $REPR \
    --exp_prefix allRuns \
    --join_loss_pool_num $JLPN \
    --tfboard 0 --test 1 --eval_epoch_jerr 200 \
    --eval_epoch 200 \
    --num_groups $NUM_GROUPS \
    --loss_func mse \
    --normalization_type mscn \
    --result_dir new_results_noval \
    --optimizer_name adamw \
    --weight_decay 0.1 \
    --use_val_set 0 \
    --weight_decay 0.01 \
    --avg_jl_num_last 20 \
    --priority_normalize_type flow4
done
