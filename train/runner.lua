
package.path = package.path .. "/home/sbt-voronova-id/traindata/train"
package.cpath = package.cpath .. "/home/sbt-voronova-id/traindata/train"
require 'pl'
require 'trepl'
require 'torch'
require 'image'
require 'gnuplot'

local config = require 'config'

torch.setdefaulttensortype('torch.DoubleTensor')

local data  = require 'data'
local train = require 'train'
local test = require 'test'

  plotting = {
    valids = {},
    epoch_ind = 0,
  }

local meta = {}
function meta.__call(t, ind)
  return t[ind][1], t[ind][2], t[ind][3]
end
setmetatable(plotting.valids,meta)

local k = 1
while k <= config.epochnm do
   plotting.epoch_ind = k;
   plotting.valids[plotting.epoch_ind] = {}
   plotting.valids[plotting.epoch_ind][1] = k
   train(trainData)
   test(testData)
   k = k + 1
end

if config.with_plotting then
  local dataf = io.open(config.data_file_path, 'w')
  for i = 1, plotting.epoch_ind do
   dataf:write(string.format('%d %f %f\n', plotting.valids(i)))
  end
  dataf:close()
  local plotv = require 'plotv'
  plotv.plotv(config.data_file_path)
end
