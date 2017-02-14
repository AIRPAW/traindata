
--package.path = package.path .. "~/torch/install/lib/luarocks/rocks"
--package.cpath = package.cpath .. "~/torch/install/lib/luarocks/rocks"
require 'torch'
require 'image'

local config = require 'config'
traintDir = config.pathToImages

local num = config.numImages
local size = config.imagesSize
local categories = config.categories
local trsize = #categories
local channels = config.channels
local img = torch.Tensor(num*trsize,channels,size.y,size.x)
local labels = torch.Tensor(num*trsize)
local trainPortion = config.trainPortion

for i = 1,#categories do
  local index = (i-1)*num
  local name = traintDir .. categories[i] .. "/"
  for j = 1, num do
    img[index+j] = image.load(name ..categories[i] ..j..".jpg")
    labels[index+j] = i
  end
end

local toMix = torch.randperm(labels:size()[1])
local trainSize = math.floor(toMix:size()[1]*trainPortion)
local testSize = toMix:size()[1] - trainSize

trainData = {
  img = torch.Tensor(trainSize, channels, size.y,size.x),
  labels = torch.Tensor(trainSize),
  size = function() return trainSize end
}

testData = {
  img = torch.Tensor(testSize, channels, size.y,size.x),
  labels = torch.Tensor(testSize),
  size = function() return testSize end
}

for i = 1, trainSize do
  trainData.img[i] = img[toMix[i]]:clone()
  trainData.labels[i] = labels[toMix[i]]
end

for i = 1, testSize do
  testData.img[i] = img[toMix[i + trainSize]]:clone()
  testData.labels[i] = labels[toMix[i + trainSize]]
end

print("dataset loaded")

return {
  trainData,
  testData,
  trsize
}
