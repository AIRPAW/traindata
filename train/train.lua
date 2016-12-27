
--package.path = package.path .. "~/torch/install/lib/luarocks/rocks"
--package.cpath = package.cpath .. "~/torch/install/lib/luarocks/rocks"
require 'torch'
require 'optim'
require 'xlua'

local t = require 'model'
local model = t.model
local fwmodel = t.model
local loss = t.loss

local optimState = {
   learningRate = config.learningRate,
   momentum = config.momentum,
   weightDecay = config.weightDecay,
   learningRateDecay = config.learningRateDecay
}

function nilling(module)
   module.gradBias   = nil
   if module.finput then module.finput = torch.Tensor() end
   module.gradWeight = nil
   module.output     = torch.Tensor()
   if module.fgradInput then module.fgradInput = torch.Tensor() end
   module.gradInput  = nil
end

function netLighter(network)
   nilling(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netLighter(a)
      end
   end
end

local x = torch.Tensor(config.batchSize,trainData.img:size(2),
         trainData.img:size(3), trainData.img:size(4))
local yt = torch.Tensor(config.batchSize)
local confusion = optim.ConfusionMatrix(category)
local epoch

local w,dE_dw = model:getParameters()

local function train(TrainData)

   epoch = epoch or 1

   local time = sys.clock()

   local shuffle = torch.randperm(TrainData:size())
   print(sys.COLORS.green .. '==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   for t = 1,TrainData:size(),config.batchSize do
      xlua.progress(t, TrainData:size())
      collectgarbage()

      if (t + config.batchSize - 1) > TrainData:size() then
         break
      end

      local idx = 1
      for i = t,t+config.batchSize-1 do
         x[idx] = TrainData.img[shuffle[i]]
         yt[idx] = TrainData.labels[shuffle[i]]
         idx = idx + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()

         -- evaluate function for complete mini batch
         local y = model:forward(x)
         --print("y size = " .. y:size()[1])
         local E = loss:forward(y,yt)

         -- estimate df/dW
         local dE_dy = loss:backward(y,yt)
         model:backward(x,dE_dy)

         -- update confusion
         for i = 1,config.batchSize do
            confusion:add(y[i],yt[i])
         end

         -- return f and df/dX
         return E,dE_dw
      end

      -- optimize on current mini-batch
      optim.sgd(eval_E, w, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / TrainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   print(confusion)
   if with_plotting then
     plotting.valids_train[plotting.epoch_ind] = confusion.totalValid;
   end

   -- save/log current net
   local filename = config.save
   os.execute('mkdir -p ' .. sys.dirname(config.save .. 'models'))
   model1 = model:clone()
  -- netLighter(model1)
   torch.save(filename .. 'models', model1)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end

-- Export:
return train
