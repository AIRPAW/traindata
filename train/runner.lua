
package.path = package.path .. "~/working/traindata/train"
package.cpath = package.cpath .. "~/working/traindata/train"
require 'pl'
require 'trepl'
require 'torch'
require 'image'

config = {
  batchSize         = 2,
  momentum          = 0,
  learningRate      = 1e-3,
  weightDecay       = 1e-5,
  learningRateDecay = 1e-7,
  save              = '/home/uml/working/traindata/models/',
  epochnm           = 150
}

local data  = require 'data'
local train = require 'train'
local train = require 'test'

local k = 1
while k < config.epochnm do
   train(trainData)
   test(testData)
   k = k + 1
end
