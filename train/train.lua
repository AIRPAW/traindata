
--package.path = package.path .. "~/torch/install/lib/luarocks/rocks"
--package.cpath = package.cpath .. "~/torch/install/lib/luarocks/rocks"
require 'torch'
require 'optim'
require 'xlua'
require 'nn'
local config = require 'config'
local t = require 'mScreenSeg'
local model  = t.model
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

local x = torch.Tensor(config.batchSize,config.channels,
         config.imagesSize.y, config.imagesSize.x)
local yt = torch.Tensor(config.batchSize, config.channels,
         config.imagesSize.y, config.imagesSize.x)
local lossPrint = {}

local epoch

local w,dE_dw = model:getParameters()

local function train(TrainData)
   epoch = epoch or 1

   local time = sys.clock()

   local shuffle = torch.randperm(TrainData:size())
   print(sys.COLORS.green .. '==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   local Eglob = 0
   for t = 1,TrainData:size(),config.batchSize do
      xlua.progress(t, TrainData:size())
      collectgarbage(collect)

      if (t + config.batchSize - 1) > TrainData:size() then
         break
      end

      local idx = 1
      for i = t,t+config.batchSize-1 do
         x[idx] = TrainData.img[shuffle[i]]
         yt[idx] = TrainData.marks[shuffle[i]]
         idx = idx + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()

         -- evaluate function for complete mini batch
         local y = model:forward(x)
         local E = loss:forward(y,yt)
         Eglob =  Eglob + E
         -- estimate df/dW
         local dE_dy = loss:backward(y,yt)
         model:backward(x,dE_dy)

         -- return f and df/dX
         return E,dE_dw
      end

      -- optimize on current mini-batch
      optim.sgd(eval_E, w, optimState);
   end

   -- time taken
   time = sys.clock() - time
   time = time / TrainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   Eglob = Eglob/(math.floor(TrainData:size()/config.batchSize))
   table.insert(lossPrint, Eglob)
   print('E = ' .. Eglob)
   if config.with_plotting then
     plotting.valids[plotting.epoch_ind][2] = Eglob;
   end

   -- save/log current net
     if epoch >= config.epochnm then
        torch.save('lossTrain.dat', lossPrint)
     end
     print('start saving...')
     local filename = config.modelPath
     os.execute('mkdir -p ' .. sys.dirname(config.modelPath))
     model:clearState()
     msave = model:clone()
     netLighter(msave)
     torch.save(filename .. 'model.t7', msave)
     msave = nil
   -- next epoch
   epoch = epoch + 1
end

-- Export:
return train
