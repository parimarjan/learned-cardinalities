
#DECAYS=(1.0 0.1)
DECAYS=(1.0)
LRS=(0.00001 0.0001)
#NORM_FLS=(1 0)
NORM_FLS=(0)
COST_MODELS=("mysql_rc2")

for j in "${!LRS[@]}";
do
  for k in "${!NORM_FLS[@]}";
  do
    for l in "${!DECAYS[@]}";
    do
        for i in "${!COST_MODELS[@]}";
        do
				CMD="bash run_all_default.sh flow_loss2 ${DECAYS[$l]} \
        ${LRS[$j]} \
        ${COST_MODELS[$i]} \
        ${NORM_FLS[$k]}"
        echo $CMD
        eval $CMD
			done
		done
	done
done

