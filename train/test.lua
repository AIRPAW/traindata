require 'torch'
require 'optim'
require 'xlua'

local config = require 'config'
local t = require 'mScreenSeg'
local model = t.model
local loss = t.loss

--local testLogger = optim.Logger(paths.concat(config.save, 'test.log'))

local x = torch.Tensor(config.batchSize,config.channels,
         config.imagesSize.y, config.imagesSize.x)
local yt = torch.Tensor(config.batchSize, config.channels,
         config.imagesSize.y, config.imagesSize.x)

function test(TestData)

   local time = sys.clock()
   local Eglob = 0
   print(sys.COLORS.red .. '==> testing on test set:')
   for t = 1,TestData:size(),config.batchSize do

      xlua.progress(t, TestData:size())

      if (t + config.batchSize - 1) > TestData:size() then
         break
      end

      local idx = 1
      for i = t,t+config.batchSize-1 do
         inputs[idx] = TestData.img[i]
         targets[idx] = TestData.marks[i]
         idx = idx + 1
      end

      local preds = model:forward(inputs)
      for i = 1,config.batchSize do
          E = loss:forward(prdes[i],targets[i])
          print('E = ' .. E )
          Eglob = Eglob + E
      end
   end

   time = sys.clock() - time
   time = time / TestData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   Eglob = Eglob/(math.floor(TrainData:size()/config.batchSize))

   if config.with_plotting then
     plotting.valids[plotting.epoch_ind][3] = Eglob;
   end
end

return test
