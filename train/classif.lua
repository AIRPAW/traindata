
-- package.path = package.path .. "~/working/traindata/train"
-- package.cpath = package.cpath .. "~/working/traindata/train"
require 'torch'
require 'optim'
require 'xlua'

model_file = '/home/uml/working/traindata/models/model.net'

local t = torch.load(model_file)
local model = t.model
local loss = t.loss

local confusion = optim.ConfusionMatrix(category)

--local testLogger = optim.Logger(paths.concat(config.save, 'test.log'))

local inputs = torch.Tensor(config.batchSize,testData.img:size(2),
         testData.img:size(3), testData.img:size(4))
local targets = torch.Tensor(config.batchSize)

function test(TestData)

   local time = sys.clock()

   print(sys.COLORS.red .. '==> testing on test set:')
   for t = 1,TestData:size(),config.batchSize do

      xlua.progress(t, TestData:size())

      if (t + config.batchSize - 1) > TestData:size() then
         break
      end

      local idx = 1
      for i = t,t+config.batchSize-1 do
         inputs[idx] = TestData.img[i]
         targets[idx] = TestData.labels[i]
         idx = idx + 1
      end

      local preds = model:forward(inputs)

      for i = 1,config.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   time = sys.clock() - time
   time = time / TestData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   print(confusion)

  --  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  --  if config.plot then
  --     testLogger:style{['% mean class accuracy (test set)'] = '-'}
  --     testLogger:plot()
  --  end
   confusion:zero()

end

return test
