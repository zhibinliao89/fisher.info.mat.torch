--  This script is modified from: 
--  https://github.com/jhkim89/PyramidNet
--  and:
--  https://github.com/facebook/fb.resnet.torch
-- 
--  The residual block is modified to have the following structure:
--  input->conv->bn-relu->conv->relu->+->
--        |                           |
--        ----------------------------|
--  ** the pyramid structure is not used.
-- 
--  ************************************************************************
--  Notes from original script:
--  Implementation of "Deep Pyramidal Residual Networks" 

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************

local nn = require 'nn'
require 'cunn'

local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling

require 'patch/SpatialConvolutionFullGrad'
require 'patch/BatchNormalizationFullGrad'
require 'patch/SpatialBatchNormalizationFullGrad'
require 'patch/LinearFullGrad'

local Convolution = cudnn.SpatialConvolutionFullGrad
local Linear = nn.LinearFullGrad
local SBatchNorm = nn.SpatialBatchNormalizationFullGrad
local linearTypeName = 'nn.LinearFullGrad'

local function createModel(opt)
   local depth = opt.depth
   local iChannels
   local alpha = 16
   local function round(x)
      return math.floor(x+0.5)
   end

   local function shortcut(nInputPlane, nOutputPlane, stride)
      -- Strided, zero-padded identity shortcut
      local short = nn.Sequential()
      if stride == 2 then
         short:add(nn.SpatialAveragePooling(2, 2, 2, 2))
      end
      if nInputPlane ~= nOutputPlane then
         short:add(nn.Padding(1, (nOutputPlane - nInputPlane), 3))
      else
	      short:add(nn.Identity())
      end
      -- if nInputPlane ~= nOutputPlane then
      --    short:add(SBatchNorm(nInputPlane))
      --    short:add(ReLU(true))
      --    short:add(Convolution(nInputPlane,nOutputPlane,3,3,stride,stride,1,1))
      -- else
      --    short:add(nn.Identity())
      -- end

      return short
   end

   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      -- print(nInputPlane ..'/' ..n)
       
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))
     
      -- return s
      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
   end

   -- Creates $count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      if count < 1 then
        return s
      end
      for i=1,count do
         s:add(block(features, stride))
      end
      return s
   end

   local model = nn.Sequential()

   local n = (depth - 2) / 6
   iChannels = 16
   local startChannel = 16
   addChannel = alpha/n
   print(' | CBRCB-ResNet-' .. depth .. ' CIFAR-10')

   model:add(Convolution(3,16,3,3,1,1,1,1))
   model:add(SBatchNorm(iChannels))

   -- 16 channels
   startChannel = 16
   model:add(layer(basicblock, round(startChannel), 1, 1, 1))
   for i=2,n do 
      model:add(layer(basicblock, round(startChannel), 1, 1, 1))
   end
   -- 32 channels
   startChannel = 32
   model:add(layer(basicblock, round(startChannel), 1, 2, 1))
   for i=2,n do 
      model:add(layer(basicblock, round(startChannel), 1, 1, 1))
   end
   -- 64 channels
   startChannel = 64
   model:add(layer(basicblock, round(startChannel), 1, 2, 1))
   for i=2,n do 
      model:add(layer(basicblock, round(startChannel), 1, 1, 1))
   end
   model:add(nn.Copy(nil, nil, true))
   model:add(SBatchNorm(iChannels))
   model:add(ReLU(true))
   model:add(Avg(8, 8, 1, 1))
   model:add(nn.View(iChannels):setNumInputDims(3))
   if opt.dataset == 'cifar10' then
      model:add(Linear(iChannels, 10))
   elseif opt.dataset == 'cifar100' then
      model:add(Linear(iChannels, 100))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   ConvInit('cudnn.SpatialConvolutionFullGrad')
   BNInit('nn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalizationFullGrad')
   for k,v in pairs(model:findModules(linearTypeName)) do
      v.bias:zero()
   end
   
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
