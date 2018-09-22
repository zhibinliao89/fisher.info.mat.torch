matio = require 'matio'

local function loadfile(fname)
	local fpath = 'data/'..fname..'/score.t7'
	local matfpath = 'data/'..fname..'/score.mat'
	local info = torch.load( fpath )
	info.top1 = torch.Tensor(info.top1)
	info.top5 = torch.Tensor(info.top5)
	info.epoch = torch.Tensor(info.epoch)
	info.trainLoss = torch.Tensor(info.trainLoss)
	info.testLoss = torch.Tensor(info.testLoss)
	if info.epochTime then
		info.epochTime = torch.Tensor(info.epochTime)
	end
	local trainCond = {}
	for i, v in ipairs(info.trainCond) do
		trainCond['e'..tostring(i)] = v
	end
	info.trainCond = trainCond
	print(info)
	matio.save(matfpath, info)
end

fnames = {}

  for i=1, #arg do
    table.insert(fnames, arg[i])
  end
  
print(fnames)

for i=1,#fnames do
 	loadfile(fnames[i])
end



