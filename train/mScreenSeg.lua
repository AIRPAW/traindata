
--package.path = package.path .. "~/torch/install/lib/luarocks/rocks"
--package.cpath = package.cpath .. "~/torch/install/lib/luarocks/rocks"
require 'torch'
require 'image'
require 'nn'
require 'nngraph'
local config = require 'config'
local pooling = nn.SpatialMaxPooling(2, 2, 2, 2)
mScreenSeg = nn.Sequential()
mScreenSeg:add(nn.SpatialConvolution(config.channels, 16, 25, 25))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(pooling)

mScreenSeg:add(nn.SpatialConvolution(16, 32, 15, 15))
mScreenSeg:add(nn.ReLU())
-- mScreenSeg:add(nn.SpatialMaxPooling(2, 2, 2, 2))

mScreenSeg:add(nn.SpatialConvolution(32, 64, 5, 5))
mScreenSeg:add(nn.ReLU())
-- mScreenSeg:add(nn.SpatialMaxPooling(2, 2, 2, 2))

mScreenSeg:add(nn.SpatialFullConvolution(64, 32, 5, 5))

mScreenSeg:add(nn.SpatialFullConvolution(32, 16, 15, 15))
mScreenSeg:add(nn.SpatialMaxUnpooling(pooling))
mScreenSeg:add(nn.SpatialFullConvolution(16, config.channels, 25, 25))

mScreenSeg:add(nn.SpatialSoftMax())

loss = nn.SmoothL1Criterion()

print("Model data:")
print(mScreenSeg)

return {
  model = mScreenSeg,
  loss = loss
}
