
--package.path = package.path .. "~/torch/install/lib/luarocks/rocks"
--package.cpath = package.cpath .. "~/torch/install/lib/luarocks/rocks"
require 'torch'
require 'image'
require 'nn'

model = nn.Sequential()
model:add(nn.SpatialConvolution(1, 16, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model:add(nn.Dropout(0.2))

model:add(nn.SpatialConvolution(16, 32, 5, 5))
model:add(nn.ReLU()) --
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model:add(nn.Dropout(0.2))

model:add(nn.View(4*4*32))
model:add(nn.Linear(4*4*32, 64))
model:add(nn.ReLU()) --
model:add(nn.Dropout(0.2))
model:add(nn.Linear(64, 20))
model:add(nn.ReLU())
model:add(nn.Linear(20, 3))
model:add(nn.LogSoftMax())

loss = nn.ClassNLLCriterion()

print("Model data:")
print(model)

return {
  model = model,
  loss = loss
}
