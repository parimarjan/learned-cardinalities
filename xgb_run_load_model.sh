MODEL_DIR=/home/pari/learned-cardinalities/all_results/xgb/default_config/nested_loop_index7-XGBoost-0.0--D0.1-115

CMD="time python3 main.py --algs xgboost --model_dir $MODEL_DIR"
echo $CMD
eval $CMD
