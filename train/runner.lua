
package.path = package.path .. "~/working/traindata/train"
package.cpath = package.cpath .. "~/working/traindata/train"
require 'pl'
require 'trepl'
require 'torch'
require 'image'
require 'gnuplot'

config = {
  batchSize         = 3,
  momentum          = 0,
  learningRate      = 1e-2,
  weightDecay       = 1e-5,
  learningRateDecay = 1e-7,
  save              = '/home/uml/working/traindata/models/',
  epochnm           = 20,
  with_plotting     = true
}

torch.setdefaulttensortype('torch.DoubleTensor')

local data  = require 'data'
local train = require 'train'
local test = require 'test'

if with_plotting then
  plotting = {

    x = torch.linspace(1,config.epochnm),
    valids_train = torch.Tensor(3, config.epochnm),
    valids_test = torch.Tensor(config.epochnm),
    epoch_ind = 0
  }
end

local k = 1
while k <= config.epochnm do
   plotting.epoch_ind = k;
   train(trainData)
   test(testData)
   k = k + 1
end
gnuplot.grid(true)
gnuplot.plotflush();
gnuplot.plot({'train',plotting.valids_train,'~'}, {'test', plotting.valids_test,'~'})
