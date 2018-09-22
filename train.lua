--  This script is modifed from:
--  https://github.com/liuzhuang13/DenseNet
--  and:
--  https://github.com/facebook/fb.resnet.torch
--  
--  Original copyright notes:
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
   
   if self.opt.enableJacobian then
      self.Jacobian = nil
      self.dataSave =  nil
      self.cond = 0
      self.condAvg = 0
      self.gradNorm = 0
      self.gradNormAvg = 0
      self.VarCond = 0
      self.rho = 0.9
   end
end

function Trainer.getParametersByClass(model, typename)
    -- insert items from table $from to table $to or item $from to table $to
    local function tinsert2(to, from, value)

        if type(from) == 'table' then
            for i=1,#from do
                tinsert2(to,from[i], value)
            end
        else
         local v = torch.Tensor(from:size())
         if value == 1 then
            table.insert(to, v:fill(1.))
         elseif value == 0 then
            table.insert(to, v:zero())
            end
        end
    end
    
    -- insert items from table $from to table $to or item $from to table $to
    local function tinsert(to, from)
        if type(from) == 'table' then
            for i=1,#from do
                tinsert(to,from[i])
            end
        else
            table.insert(to,from)
        end
    end

    local w = {}
    if model.modules then
       for i=1,#model.modules do
            local mw = Trainer.getParametersByClass(model.modules[i], typename)
            tinsert(w, mw)
       end
    else
      
      local mod_type = torch.typename(model)
      local mw = model:parameters()
      
      if mw then
         if mod_type == typename then
            tinsert2(w, mw, 1)
         else
            tinsert2(w, mw, 0)
         end
      end
    end

    return w
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local epochTimer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end
  
   -- get the saved Jacobian matrices of each layer with trainable parameters
   -- and concatenate all these matrices
   local function getJacobian(batchSize)
      local tab = {}
      local ls = torch.LongStorage{batchSize, -1}

      for k, v in pairs(self.model:findModules('cudnn.SpatialConvolutionFullGrad')) do
         table.insert(tab, v.gradWeightFull:view(ls))
         if v.gradBiasFull ~= nil then
            table.insert(tab, v.gradBiasFull:view(ls))
         end
      end

      for k, v in pairs(self.model:findModules('nn.SpatialBatchNormalizationFullGrad')) do
         if v.gradWeightFull ~= nil then
            table.insert(tab, v.gradWeightFull:view(ls))
            table.insert(tab, v.gradBiasFull:view(ls))
         end
      end

      for k, v in pairs(self.model:findModules('nn.LinearFullGrad')) do
         table.insert(tab, v.gradWeightFull:view(ls))
         table.insert(tab, v.gradBiasFull:view(ls))
      end

      return torch.cat(tab, 2)
   end
   
   -- computing the Jacobian for every mini-batch can be slow, instead this function is used to 
   -- control whether or not to compute the Jocobian for this mini-batch.
   local function setGetFullGrad(b)
      for k, v in pairs(self.model:findModules('cudnn.SpatialConvolutionFullGrad')) do
         v.getFullGrad = b
      end

      for k, v in pairs(self.model:findModules('nn.SpatialBatchNormalizationFullGrad')) do
         v.getFullGrad = b
      end

      for k, v in pairs(self.model:findModules('nn.LinearFullGrad')) do
         v.getFullGrad = b
      end
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0
   local verboseInterval = self.opt.jacobianSamplingInterval;
   self.dataSave = nil
      
   print('=> Training epoch # ' .. epoch)
   print('  # of model params: ' .. self.gradParams:size(1))
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      if self.opt.enableJacobian then
         if epoch <= self.opt.numJacobianEvaluatingEpochs or math.fmod(n, self.opt.jacobianSamplingInterval) == 1 then
            setGetFullGrad(true)
         else
            setGetFullGrad(false)
         end
      end

      -- training forward phase
      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      -- loss
      local loss = self.criterion:forward(self.model.output, self.target)

      -- backward phase
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      --exam empirical fisher
      if n == 1 then
         -- dataSave is a matrix which stores the followings: 
         -- 1. moving average of condition number of the Jaocobian (runtime-averaged), 
         -- 2. maximum eigenvalue of the Jacobian
         -- 3. minimum eigenvalue of the Jacobian
         -- 4->batchSize+3. all eigenvalues of the Jacobian
         -- batchSize+4. gradient norm
         -- in each row
         self.dataSave = torch.Tensor(trainSize,3+batchSize+1):zero()
      end

      if self.opt.enableJacobian then
         if (epoch <= self.opt.numJacobianEvaluatingEpochs or math.fmod(n, self.opt.jacobianSamplingInterval) == 1) then
            -- the inidividual gradient vector in the Jacobian is already normalized by the batch size 
            -- by underlying torch nn libraries so the correct Jacobian needs to be scaled back.
            self.Jacobian = getJacobian(batchSize) * batchSize 

            -- Jacobian shape:  batch_size X #model_parameters
            if n == 1 then
               print( 'Eyeball Checking: Jacobian #columns: ' .. self.Jacobian:size(2)..' vs Gardient #elements: '..self.gradParams:size(1) )
            end

            local JJT = self.Jacobian * self.Jacobian:t() --eig(JJ^T) == eig(J^TJ)          
            self.gradNorm = torch.norm(self.gradParams) -- the norm of the gradient
            self.Jacobian = torch.Tensor()

            local e = torch.symeig(JJT:float(), 'N')

            -- JJ^T is symmetric so it should not have any negative eigenvalues
            e = torch.cmax(e, 0.0) 
            -- compute the square root of eigenvalues of JJ^T, which are the eigenvalues of the Jacobian
            e = torch.sqrt(e)
            -- this may give NaN as the smallest eigenvalue can be 0
            self.cond = torch.max(e)/torch.min(e)
            -- lr 
            self.dataSave[{n,2}] = self.optimState.learningRate
            -- batch size 
            self.dataSave[{n,3}] = batchSize
            -- all eigenvalues
            self.dataSave[{n,{4,batchSize+3}}]:copy(e)  
            -- gradient norm
            self.dataSave[{n,-1}] = self.gradNorm    
            ---------------------------------------------------------------------------------------------------
            -- the moving-averaged condition number, for runtime eyeball checking purpose
            if n == 1 and epoch == 1 then
               self.condAvg = self.cond
               self.gradNormAvg = self.gradNorm
            else
               self.condAvg = self.rho*self.condAvg + (1-self.rho)*self.cond
               self.gradNormAvg = self.rho*self.gradNormAvg + (1-self.rho)*self.gradNorm
            end

            self.dataSave[{n,1}] = self.condAvg

            if n == 1 or math.fmod(n, self.opt.jacobianSamplingInterval) == 1 then
               print((' Norm %0.2f  Frob %0.2f  Cond %0.2f  EMax %0.2f  EMin %0.2f'):format(self.gradNorm, torch.sum(self.dataSave[{n,{4,batchSize+3}}]), self.dataSave[{n,1}], torch.max(e), torch.min(e)))
            end

         end
      end
      optim.sgd(feval, self.params, self.optimState)

      local top1, top5 = self:computeScore(output, sample.target, 1)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize
      
      if math.fmod(n, verboseInterval) == 0 or n == trainSize then
         print((' Epoch: [%3d][%3d/%3d]  T:%.3f  D:%.3f  Obj %1.4f(%1.4f)  t1 %7.3f(%7.3f)  t5 %7.3f(%7.3f)'):format(
            epoch, n, trainSize, timer:time().real, dataTime, loss, lossSum / N, top1, top1Sum / N, top5, top5Sum / N))
      end
      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   local dataSaveOut = torch.Tensor(self.dataSave:size()):copy(self.dataSave)
   local epochTime = epochTimer:time().real
   print(('Current time: %s. Training uses %f scondAvgs\n'):format(os.date('%c'), epochTime))

   return top1Sum / N, top5Sum / N, lossSum / N, dataSaveOut, epochTime
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local epochTimer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)

      local top1, top5 = self:computeScore(output, sample.target, nCrops)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize
      if math.fmod(n, 50) == 0 or n == size then
         print((' Test: [%d][%3d/%3d]  T:%.3f  D:%.3f  Obj %1.4f(%1.4f)  t1 %7.3f(%7.3f)  t5 %7.3f(%7.3f)'):format(
            epoch, n, size, timer:time().real, dataTime, loss, lossSum / N, top1, top1Sum / N, top5, top5Sum / N))
      end
      timer:reset()
      dataTimer:reset()
   end
   self.model:training()
   local epochTime = epochTimer:time().real
   print(('Current time: %s. Test uses %f scondAvgs\n'):format(os.date('%c'), epochTime))

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))

   return top1Sum / N, top5Sum / N, lossSum / N, epochTime
end

function Trainer:computeScore(output, target, nCrops)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

local function getCudaTensorType(tensorType)
  if tensorType == 'torch.CudaHalfTensor' then
     return cutorch.createCudaHostHalfTensor()
  elseif tensorType == 'torch.CudaDoubleTensor' then
    return cutorch.createCudaHostDoubleTensor()
  else
     return cutorch.createCudaHostTensor()
  end
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch[self.opt.tensorType:match('torch.(%a+)')]()
      or getCudaTensorType(self.opt.tensorType))
   self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   local decay = 0

   -- Training schedule
   if self.opt.learningRateSchedule == 1 then
      decay = epoch >= 241 and 2 or epoch >= 161 and 1 or 0     
   end

   print(' Learning Rate Schedule No.'..self.opt.learningRateSchedule)
   local rate = self.opt.LR * math.pow(0.1, decay)
   print((' Using Learning Rate: %0.4f'):format(rate))
   return rate
end


return M.Trainer
