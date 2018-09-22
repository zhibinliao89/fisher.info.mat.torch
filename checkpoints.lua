--  This script is modifed from:
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
local checkpoint = {}

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))

   return latest, optimState
end

function checkpoint.any(opt)
   if opt.resume == 'none' then
      return nil
   end

   local epoch = opt.epochNumber
   local modelFile = 'model_' .. epoch .. '.t7'
   local optimFile = 'optimState_' .. epoch .. '.t7'

   latest = {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
   }
   -- print(latest)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))
   -- print(optimState)
   return latest, optimState
end

function checkpoint.save(epoch, model, optimState, isBestModel, opt, trainTop1, trainTop5, trainLoss, trainCond, trainEpochTime, testTop1, testTop5, testLoss, testEpochTime)
   -- don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- create a clean copy on the CPU without modifying the original network
   model = deepCopy(model):float():clearState()

   local modelFile = 'model_' .. epoch .. '.t7'
   local optimFile = 'optimState_' .. epoch .. '.t7'
   local scoreFile = 'score' .. '.t7'

   local scorePath = paths.concat(opt.save, scoreFile)
   local score
   print(epoch)
   if paths.filep(scorePath) and epoch ~= 1 then
      score = torch.load(scorePath)
   else
      score = {top1 = {}, top5 = {}, trainLoss = {}, testLoss = {}, epoch = {}, trainCond = {}, epochTime = {}}
   end

   table.insert(score.top1, {trainTop1, testTop1})
   table.insert(score.top5, {trainTop5, testTop5})
   table.insert(score.epochTime, {trainEpochTime, testEpochTime})
   table.insert(score.trainLoss, trainLoss)
   table.insert(score.testLoss, testLoss)
   table.insert(score.epoch, epoch)
   if trainCond ~= nil then
      table.insert(score.trainCond, trainCond)
   end
   -- print(trainCond)
   torch.save(paths.concat(opt.save, modelFile), model)
   torch.save(paths.concat(opt.save, optimFile), optimState)
   torch.save(paths.concat(opt.save, 'latest.t7'), {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
   })

   if isBestModel then
      torch.save(paths.concat(opt.save, 'model_best.t7'), model)
   end
   torch.save(paths.concat(opt.save, scoreFile), score)
end



function checkpoint.saveScore(epoch, opt, trainTop1, trainTop5, trainLoss, testTop1, testTop5, testLoss)
  
   local scoreFile = 'score' .. '.t7'

   local scorePath = paths.concat(opt.save, scoreFile)
   local score
   -- print(epoch)
   if paths.filep(scorePath) and epoch ~= 1 then
      score = torch.load(scorePath)
   else
      score = {top1 = {}, top5 = {}, trainLoss = {}, testLoss = {}, epoch = {}}
   end

   table.insert(score.top1, {trainTop1, testTop1})
   table.insert(score.top5, {trainTop5, testTop5})
   table.insert(score.trainLoss, trainLoss)
   table.insert(score.testLoss, testLoss)
   table.insert(score.epoch, epoch)


   torch.save(paths.concat(opt.save, scoreFile), score)
end

return checkpoint