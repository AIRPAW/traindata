
--package.path = package.path .. "~/torch/install/lib/luarocks/rocks"
--package.cpath = package.cpath .. "~/torch/install/lib/luarocks/rocks"
require 'torch'
require 'image'
require 'nn'
require 'nngraph'
local config = require 'config'
local pooling2 = nn.SpatialMaxPooling(2, 2, 2, 2)
local pooling1 = nn.SpatialMaxPooling(2, 2, 1, 1)
local pooling16 = nn.SpatialMaxPooling(16, 16, 2, 2)
local pooling32 = nn.SpatialMaxPooling(32, 32, 2, 2)

mScreenSeg = nn.Sequential()
--encoder
mScreenSeg:add(nn.SpatialConvolution(3, 5, 3, 3))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(5,1e-05 ,1e-03))

mScreenSeg:add(pooling1)
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(5,1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialConvolution(5, 8, 3, 3))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(8, 1e-05 ,1e-03))

mScreenSeg:add(pooling1)
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(8, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialConvolution(8, 16, 3, 3))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(16, 1e-05 ,1e-03))

mScreenSeg:add(pooling16)
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(16, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialConvolution(16, 32, 3, 3))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(32, 1e-05 ,1e-03))

mScreenSeg:add(pooling2)
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(32, 1e-05 ,1e-03))

mScreenSeg:add(pooling32)
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(64, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialConvolution(32, 64, 5, 5))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(64, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialConvolution(64, 128, 5, 5))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(pooling32)
mScreenSeg:add(nn.SpatialBatchNormalization(128, 1e-05 ,1e-03))

--decoder
mScreenSeg:add(nn.SpatialFullConvolution(128, 64, 5, 5))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialMaxUnpooling(pooling32))
mScreenSeg:add(nn.SpatialBatchNormalization(64, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialFullConvolution(64, 32, 5, 5))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(32,1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialMaxUnpooling(pooling32))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(64, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialMaxUnpooling(pooling2))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(32, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialFullConvolution(32, 16, 3, 3))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(16, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialMaxUnpooling(pooling16))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(16, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialFullConvolution(16, 8, 3, 3))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(16, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialMaxUnpooling(pooling1))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(16, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialFullConvolution(8, 5, 3, 3))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(5, 1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialMaxUnpooling(pooling1))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(5,1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialFullConvolution(5, 3, 3, 3))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(3,1e-05 ,1e-03))

mScreenSeg:add(nn.SpatialFullConvolution(3, 1, 3, 3))
mScreenSeg:add(nn.ReLU())
mScreenSeg:add(nn.SpatialBatchNormalization(1,1e-05 ,1e-03))
----
mScreenSeg:add(nn.LogSoftMax())

loss = nn.ClassNLLCriterion()

print("Model data:")
print(mScreenSeg)
return {
 model = mScreenSeg,
 loss = loss
}
