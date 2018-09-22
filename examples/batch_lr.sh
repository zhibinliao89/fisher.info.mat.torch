export CUDA_VISIBLE_DEVICES=$1
echo 'USING GPU: '$CUDA_VISIBLE_DEVICES
exp_base=0
net_type=$2
data_set=$3
lr=$4
batch_size=$5
extra='_size'$batch_size'_lr'$lr'_example'
echo 'dataset: '$data_set
echo 'network type: '$net_type
echo 'learning rate: '$lr
echo 'batch size: '$batch_size
echo 'folder_suffix: '$extra
for exp_no in {1..1}
	do
	th main.lua -dataset $data_set \
				-depth 110 \
				-nEpochs 320 \
				-LR $lr \
				-netType $net_type \
				-batchSize $batch_size \
				-shareGradInput true \
	            -enableJacobian true \
	            -numJacobianEvaluatingEpochs 0 \
	            -learningRateSchedule 1 \
	            -save data/$data_set/$net_type$extra/$((exp_no+exp_base)) \
	            -resume data/$data_set/$net_type$extra/$((exp_no+exp_base))
done
