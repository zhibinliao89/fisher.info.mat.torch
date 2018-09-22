-- This script is modified from: https://github.com/liuzhuang13/DenseNet

require 'nn'
require 'cudnn'
require 'cunn'

require 'patch/SpatialConvolutionFullGrad'
require 'patch/BatchNormalizationFullGrad'
require 'patch/SpatialBatchNormalizationFullGrad'

local Convolution = cudnn.SpatialConvolutionFullGrad
local SBatchNorm = nn.SpatialBatchNormalizationFullGrad


local function ShareGradInput(module, key)
   assert(key)
   module.__shareGradInputKey = key
   return module
end

--------------------------------------------------------------------------------
-- Standard densely connected layer (memory inefficient)
--------------------------------------------------------------------------------
function DenseConnectLayerStandard(nChannels, opt)
   local net = nn.Sequential()

   net:add(ShareGradInput(SBatchNorm(nChannels), 'first'))
   net:add(cudnn.ReLU(true))   
   if opt.bottleneck then
      net:add(Convolution(nChannels, 4 * opt.growthRate, 1, 1, 1, 1, 0, 0))
      nChannels = 4 * opt.growthRate
      if opt.dropRate > 0 then net:add(nn.Dropout(opt.dropRate)) end
      net:add(SBatchNorm(nChannels))
      net:add(cudnn.ReLU(true))      
   end
   net:add(Convolution(nChannels, opt.growthRate, 3, 3, 1, 1, 1, 1))
   if opt.dropRate > 0 then net:add(nn.Dropout(opt.dropRate)) end

   return nn.Sequential()
      :add(nn.Concat(2)
         :add(nn.Identity())
         :add(net))  
end

--------------------------------------------------------------------------------
-- Customized densely connected layer (memory efficient)
--------------------------------------------------------------------------------
local DenseConnectLayerCustom, parent = torch.class('nn.DenseConnectLayerCustom', 'nn.Container')

function DenseConnectLayerCustom:__init(nChannels, opt)
   parent.__init(self)
   self.train = true
   self.opt = opt

   self.net1 = nn.Sequential()
   self.net1:add(ShareGradInput(SBatchNorm(nChannels), 'first'))
   self.net1:add(cudnn.ReLU(true))  

   self.net2 = nn.Sequential()
   if opt.bottleneck then
      self.net2:add(Convolution(nChannels, 4*opt.growthRate, 1, 1, 1, 1, 0, 0))
      nChannels = 4 * opt.growthRate
      self.net2:add(SBatchNorm(nChannels))
      self.net2:add(cudnn.ReLU(true))
   end
   self.net2:add(Convolution(nChannels, opt.growthRate, 3, 3, 1, 1, 1, 1))

   -- contiguous outputs of previous layers
   self.input_c = torch.Tensor():type(opt.tensorType) 
   -- save a copy of BatchNorm statistics before forwarding it for the second time when optMemory=4
   self.saved_bn_running_mean = torch.Tensor():type(opt.tensorType)
   self.saved_bn_running_var = torch.Tensor():type(opt.tensorType)

   self.gradInput = {}
   self.output = {}

   self.modules = {self.net1, self.net2}
end

function DenseConnectLayerCustom:updateOutput(input)

   if type(input) ~= 'table' then
      self.output[1] = input
      self.output[2] = self.net2:forward(self.net1:forward(input))
   else
      for i = 1, #input do
         self.output[i] = input[i]
      end
      torch.cat(self.input_c, input, 2)
      self.net1:forward(self.input_c)
      self.output[#input+1] = self.net2:forward(self.net1.output)
   end

   if self.opt.optMemory == 4 then
      local running_mean, running_var = self.net1:get(1).running_mean, self.net1:get(1).running_var
      self.saved_bn_running_mean:resizeAs(running_mean):copy(running_mean)
      self.saved_bn_running_var:resizeAs(running_var):copy(running_var)
   end

   return self.output
end

function DenseConnectLayerCustom:updateGradInput(input, gradOutput)

   if type(input) ~= 'table' then
      self.gradInput = gradOutput[1]
      if self.opt.optMemory == 4 then self.net1:forward(input) end
      self.net2:updateGradInput(self.net1.output, gradOutput[2])
      self.gradInput:add(self.net1:updateGradInput(input, self.net2.gradInput))
   else
      torch.cat(self.input_c, input, 2)
      if self.opt.optMemory == 4 then self.net1:forward(self.input_c) end
      self.net2:updateGradInput(self.net1.output, gradOutput[#gradOutput])
      self.net1:updateGradInput(self.input_c, self.net2.gradInput)
      local nC = 1
      for i = 1, #input do
         self.gradInput[i] = gradOutput[i]
         self.gradInput[i]:add(self.net1.gradInput:narrow(2,nC,input[i]:size(2)))
         nC = nC + input[i]:size(2)
      end
   end

   if self.opt.optMemory == 4 then
      self.net1:get(1).running_mean:copy(self.saved_bn_running_mean)
      self.net1:get(1).running_var:copy(self.saved_bn_running_var)
   end

   return self.gradInput
end

function DenseConnectLayerCustom:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.net2:accGradParameters(self.net1.output, gradOutput[#gradOutput], scale)
   if type(input) ~= 'table' then
      self.net1:accGradParameters(input, self.net2.gradInput, scale)
   else
      self.net1:accGradParameters(self.input_c, self.net2.gradInput, scale)
   end
end

function DenseConnectLayerCustom:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local lastNext = '   `-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = 'DenseConnectLayerCustom'
   str = str .. ' {' .. line .. tab .. '{input}'
   for i=1,#self.modules do
      if i == #self.modules then
         str = str .. line .. tab .. lastNext .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. '{output}'
   str = str .. line .. '}'
   return str
end
