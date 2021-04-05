
#NORM_FLS=(1 0)
#DECAYS=(0.1 1.0)
#DECAYS=(1.0)
#LRS=(0.00001)

NORM_FLS=(1)
DECAYS=(0.1 1.0)
COST_MODELS=("mysql_rc2")
LRS=(0.0001 0.00001)

for i in "${!COST_MODELS[@]}";
do
	for j in "${!LRS[@]}";
	do
		for k in "${!NORM_FLS[@]}";
		do
			for l in "${!DECAYS[@]}";
			do
				CMD="bash run_all_default_mscn.sh flow_loss2 ${DECAYS[$l]} \
        ${LRS[$j]} \
        ${COST_MODELS[$i]} \
        ${NORM_FLS[$k]}"
        echo $CMD
        eval $CMD
			done
		done
	done
done



