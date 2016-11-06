
package.path = package.path .. "~/working/traindata/train"
package.cpath = package.cpath .. "~/working/traindata/train"
require 'pl'
require 'trepl'
require 'torch'
require 'image'

config = {
  batchSize         = 3,
  momentum          = 0,
  learningRate      = 1e-3,
  weightDecay       = 1e-5,
  learningRateDecay = 1e-7,
}

local data  = require 'data'
local train = require 'train'

while true do
   train(trainData)
end
