--[[
   This file implements Batch Normalization as described in the paper:
   "Batch Normalization: Accelerating Deep Network Training
                         by Reducing Internal Covariate Shift"
                   by Sergey Ioffe, Christian Szegedy

   This implementation is useful for inputs NOT coming from convolution layers.
   For convolution layers, use nn.SpatialBatchNormalization.

   The operation implemented is:
   y =     ( x - mean(x) )
        -------------------- * gamma + beta
        standard-deviation(x)
   where gamma and beta are learnable parameters.

   The learning of gamma and beta is optional.

   Usage:
   with    learnable parameters: nn.SpatialBatchNormalization(N [,eps] [,momentum])
                                 where N = dimensionality of input
   without learnable parameters: nn.SpatialBatchNormalization(N [,eps] [,momentum], false)

   eps is a small value added to the standard-deviation to avoid divide-by-zero.
       Defaults to 1e-5

   In training time, this layer keeps a running estimate of it's computed mean and std.
   The running sum is kept with a default momentum of 0.1 (unless over-ridden)
   In test time, this running mean/std is used to normalize.
]]--
local BN, parent = torch.class('nn.SpatialBatchNormalizationFullGrad', 'nn.BatchNormalizationFullGrad')

BN.__version = 2

-- expected dimension of input
BN.nDim = 4
