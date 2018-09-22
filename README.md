Approximate Fisher Information Matrix to Characterise the Training of Deep Neural Networks
============================

This repository implements the functionality of the Jacobian matrix and the approximate Fisher Information Matrix (FIM) (a.k.a., the sample covarience matrix) calculation during the training of deep neural networks.  The Jacobian matrix can be used to calculate the FIM by: ``FIM = JJ^T``, and we can derive the eigenvalues and the condition number of the FIM to characterise the training of a network.  A detailed explaination can be found in [this paper](https://arxiv.org/).

This code extends the [torch/nn](https://github.com/torch/nn) library for computing the Jacobian for nn.linear, nn.BatchNormalization, and nn.SpatialBatchNormalization layers.  It also extends the [cudnn.torch](https://github.com/soumith/cudnn.torch) library for computing the Jacobian for cudnn.SpatialConvolution layer.  The extensions for the respective layers can be found in [patch](patch) in this format: layernameFullGrad.  Please note there are many other trainable layers, but they are not considered in this repository.

This code is built on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) for training ResNet and on [DenseNet](https://github.com/liuzhuang13/DenseNet) for DenseNet compatibility.

## Installation

Please refer to [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md) for CUDA, CuDNN, and Torch libraries installation.

## Usage

### Training A Network

Please clone this repository first and run:
```bash
source run_experiments.sh
```
for training a ResNet with learning rate 0.1 and mini-batch size 64 as an example.  Saving the Jacobian and the FIM matrices for each training iteration is unrealistic, so we only save the eigenvalues and the condition number of the FIM.

After the training of each epoch, the result will be updated in this file: ``data/cifar10/cbrcbresnet_size64_lr0.1_example/1/score.t7``.  The below is an example of the lua data table saved after 3 epochs of training:
>{  
>  epoch : DoubleTensor - size: 3  
>  trainCond :   
>    {  
>      e3 : FloatTensor - size: 782x68  
>      e2 : FloatTensor - size: 782x68  
>      e1 : FloatTensor - size: 782x68  
>    }  
>  epochTime : DoubleTensor - size: 3x2  
>  trainLoss : DoubleTensor - size: 3  
>  top5 : DoubleTensor - size: 3x2  
>  testLoss : DoubleTensor - size: 3  
>  top1 : DoubleTensor - size: 3x2  
>}. 

More specificaly, in the ``trainCond`` element of the table, each epoch will insert a tensor with the name ``e#``.  Using the first epoch as an example, the first dimension of ``trainCond.e1`` indicates the the number of iterations excuted in epoch 1, i.e., ``dim1=ceil(50000/64)=782`` iterations (50000 is the number of samples in the CIFAR-10 training set) and the second dimension is composited by ``4 + mini-batch size of the epoch``, i.e., ``dim2=64+4``.  The 64 elements are arranged as:

>e1[{n,1}]: moving average of condition number of the Jaocobian (runtime-averaged);  
>e1[{n,2}]: learning rate;  
>e1[{n,3}]: mini-batch size;  
>e1[{n,{4, -2}}]: all eigenvalues of the Jacobian;  
>e1[{n,-1}]: the norm of the gradient;

for a single iteration.  

### Porting Results to MATLAB .mat format

We provide scripts to analyze the result in MATAB.  Please run the command below to port the saved data to MATLAB ``.mat`` format:
```bash
th toMat.lua cifar10/cbrcbresnet_size64_lr0.1_example/1
```
where the first argument ``cifar10/cbrcbresnet_size64_lr0.1_example/1`` is the path to the previously trained ResNet under the folder ``data/``.  Multiple arguments can be given for batch interpretion.  The matfile is saved beside the ``.t7`` file in the same folder.

Please use MATLAB to run ``matlab/plot_single_model.m`` inside the folder ``matlab`` to visualize the result of the aforementioned ResNet model.

### Visualization of Multiple Models 

An example script of multi-model visualzation has been provided in ``matlab/plot_mulitple_models.m`` with the use of a set of pre-trained DenseNet models which were trained with various learning and mini-batch combinations.  The recorded ``score.mat`` files can be retrieved from [here](https://drive.google.com/file/d/1Wrsmw2PNDGMXCMNb09ZrwfHNfexrd_l4/view?usp=sharing).  Please place the extracted content under ``data/cifar10/`` and run ``plot_mulitple_models.m`` from ``matlab/`` folder to use this script.

## Known Issues

Please be advised that this code is intended be ran on a single GPU.  The support for computing the Jacobian with the use of multiple GPUs is not implemented.  This will limit the number of parameter of a model to be trained.

The computation of the Jacobian is a relative slow process due to computation at the torch programming layer.  A faster Jacobian computation should be achievable by direct computation at NVIDIA cuDNN programming layer but it is not implemented in this repository.