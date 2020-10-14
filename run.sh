NN_TYPE=$1
FF=0
EPOCHS=10
LR=0.0001

#bash run_all_default.sh $LR flow_loss2 $NN_TYPE 0.1 1 -1 $FF $EPOCHS 0
#bash run_all_default.sh $LR flow_loss2 mscn 0.1 1 -1 1 $EPOCHS 0
#bash run_all_default.sh $LR flow_loss2 $NN_TYPE 1.0 1 -1 $FF $EPOCHS 0
#bash run_all_default.sh 0.0005 flow_loss2 $NN_TYPE 1.0 1 -1 $FF $EPOCHS 0

bash run_all_default.sh $LR flow_loss2 $NN_TYPE 0.1 0 -1 $FF $EPOCHS 1

#bash run_all_default.sh $LR flow_loss2 $NN_TYPE 0.1 0 -1 1 $EPOCHS 0
