
export CUDA_VISIBLE_DEVICES=$1
echo 'USING GPU: '$CUDA_VISIBLE_DEVICES
exp_base=0
net_type=$2
data_set=$3
batch_size_1=$4
batch_size_2=$5
batch_size_3=$6
batch_size_4=$7
batch_size_5=$8
extra='_size'$batch_size_1'_'$batch_size_2'_'$batch_size_3'_'$batch_size_4'_'$batch_size_5'_example'
echo 'dataset: '$data_set
echo 'network type: '$net_type
echo 'folder_suffix: '$extra

for exp_no in {1..1}
	do
	th main.lua -dataset $data_set \
				-depth 110 \
				-nEpochs 64 \
				-LR 0.1 \
				-netType $net_type \
				-batchSize $batch_size_1 \
				-shareGradInput true \
	            -enableJacobian true \
	            -numJacobianEvaluatingEpochs 0 \
	            -learningRateSchedule 1 \
	            -save data/$data_set/$net_type$extra/$((exp_no+exp_base)) \
	            -resume data/$data_set/$net_type$extra/$((exp_no+exp_base))

	th main.lua -dataset $data_set \
				-depth 110 \
				-nEpochs 128 \
				-LR 0.1 \
				-netType $net_type \
				-batchSize $batch_size_2 \
				-shareGradInput true \
	            -enableJacobian true \
	            -numJacobianEvaluatingEpochs 0 \
	            -learningRateSchedule 1 \
	            -save data/$data_set/$net_type$extra/$((exp_no+exp_base)) \
	            -resume data/$data_set/$net_type$extra/$((exp_no+exp_base))            

	th main.lua -dataset $data_set \
				-depth 110 \
				-nEpochs 192 \
				-LR 0.1 \
				-netType $net_type \
				-batchSize $batch_size_3 \
				-shareGradInput true \
	            -enableJacobian true \
	            -numJacobianEvaluatingEpochs 0 \
	            -learningRateSchedule 1 \
	            -save data/$data_set/$net_type$extra/$((exp_no+exp_base)) \
	            -resume data/$data_set/$net_type$extra/$((exp_no+exp_base))            

	th main.lua -dataset $data_set \
				-depth 110 \
				-nEpochs 256 \
				-LR 0.1 \
				-netType $net_type \
				-batchSize $batch_size_4 \
				-shareGradInput true \
	            -enableJacobian true \
	            -numJacobianEvaluatingEpochs 0 \
	            -learningRateSchedule 1 \
	            -save data/$data_set/$net_type$extra/$((exp_no+exp_base)) \
	            -resume data/$data_set/$net_type$extra/$((exp_no+exp_base))        

	th main.lua -dataset $data_set \
				-depth 110 \
				-nEpochs 320 \
				-LR 0.1 \
				-netType $net_type \
				-batchSize $batch_size_5 \
				-shareGradInput true \
	            -enableJacobian true \
	            -numJacobianEvaluatingEpochs 0 \
	            -learningRateSchedule 1 \
	            -save data/$data_set/$net_type$extra/$((exp_no+exp_base)) \
	            -resume data/$data_set/$net_type$extra/$((exp_no+exp_base))        	                       
done