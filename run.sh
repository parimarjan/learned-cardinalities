
#LOSS_FUNC=$1
#DECAY=$2
#LR=$3
#MAX_EPOCHS=$4
#TRAIN_CARD_KEY=$5

bash run_wanderjoin.sh flow_loss2 1.0 0.00001 15 wanderjoin
bash run_wanderjoin.sh mse 0.1 0.0001 10 wanderjoin

#bash run_wanderjoin.sh flow_loss2 1.0 0.00001 15 actual
