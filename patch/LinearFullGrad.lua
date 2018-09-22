local LinearFullGrad, parent = torch.class('nn.LinearFullGrad', 'nn.Module')

function LinearFullGrad:__init(inputSize, outputSize, bias)
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradWeightFull = nil
   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
      self.gradBiasFull = nil
   end
   self.getFullGrad = false
   self:reset()
end

function LinearFullGrad:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function LinearFullGrad:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
      if self.bias then
         for i=1,self.bias:nElement() do
            self.bias[i] = torch.uniform(-stdv, stdv)
         end
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then self.bias:uniform(-stdv, stdv) end
   end
   return self
end

function LinearFullGrad:updateAddBuffer(input)
   local nframe = input:size(1)
   self.addBuffer = self.addBuffer or input.new()
   if self.addBuffer:nElement() ~= nframe then
      self.addBuffer:resize(nframe):fill(1)
   end
end

function LinearFullGrad:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.bias then self.output:copy(self.bias) else self.output:zero() end
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self:updateAddBuffer(input)
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearFullGrad:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function LinearFullGrad:gradFullSpacing(input)
   if self.getFullGrad then
      if self.bias then
         if self.gradBiasFull == nil then
            self.gradBiasFull = torch.Tensor(input:size(1), self.bias:size(1)):cuda():zero()
         elseif self.gradBiasFull:nDimension() == 0 then
            self.gradBiasFull = torch.Tensor(input:size(1), self.bias:size(1)):cuda():zero()
         elseif self.gradBiasFull:size(1) ~= input:size(1) then
            self.gradBiasFull = torch.Tensor(input:size(1), self.bias:size(1)):cuda():zero()
         end
      end

      if self.gradWeightFull == nil then
         self.gradWeightFull = torch.Tensor(input:size(1), self.weight:size(1), self.weight:size(2)):cuda():zero()
      elseif self.gradWeightFull:nDimension() == 0 then
         self.gradWeightFull = torch.Tensor(input:size(1), self.weight:size(1), self.weight:size(2)):cuda():zero() 
      elseif self.gradWeightFull:size(1) ~= input:size(1) then
         self.gradWeightFull = torch.Tensor(input:size(1), self.weight:size(1), self.weight:size(2)):cuda():zero()   
      end
   end
end

function LinearFullGrad:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   self:gradFullSpacing(input)

   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      if self.bias then self.gradBias:add(scale, gradOutput) end
      if self.getFullGrad then
         error('The full gradient function is not implemented for dense layer with 1 dimension')
      end
   elseif input:dim() == 2 then
      if self.getFullGrad then
         for luaIndex = 1, input:size(1) do
            local vec1 = gradOutput:narrow(1,luaIndex,1):squeeze()
            local vec2 = input:narrow(1,luaIndex,1):squeeze()
            local outprod = torch.ger(vec1, vec2):view(1, vec1:size(1), vec2:size(1))
            -- self.gradWeightFull:indexCopy(1, torch.LongTensor{luaIndex}, outprod)
            self.gradWeightFull[{luaIndex}] = outprod
         end
      end
      self.gradWeight:addmm(scale, gradOutput:t(), input)

      if self.bias then
         self:updateAddBuffer(input)
         if self.getFullGrad then
            self.gradBiasFull:copy(gradOutput)
         end
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end

end

function LinearFullGrad:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAccUpdateGradParameters(input, gradOutput, lr)
end

function LinearFullGrad:clearState()
   if self.addBuffer then self.addBuffer:set() end
   nn.utils.clear(self, 'gradWeightFull', 'gradBiasFull')
   return parent.clearState(self)
end

function LinearFullGrad:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
