GPU=0 # use GPU X
for lr in 0.1
do
	for bs in 64
	do
		source examples/batch_lr.sh $GPU cbrcbresnet cifar10 $lr $bs
	done
done

# Another example for train a models with 5 different mini-batch sizes.
# This is implemented by simply stopping a model at a certain epoch
# and resuming with another mini-batch size:

# source examples/batch_5sizes.sh $GPU densenet cifar10 16 32 64 125 256

