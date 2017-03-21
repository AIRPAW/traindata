
--package.path = package.path .. "~/torch/install/lib/luarocks/rocks"
--package.cpath = package.cpath .. "~/torch/install/lib/luarocks/rocks"
require 'torch'
require 'image'

local config = require 'config'
traintDir = config.pathToImages

local num = config.numImages
local size = config.imagesSize
local categories = config.categories
local channels = config.channels
local trainPortion = config.trainPortion

local imgStore = {}
local lablesStore = {}
local singleImgTensor = torch.Tensor(channels,size.y,size.x)

local lowerPath = config.pathToImages
local curCategory = 1
for dir in paths.iterdirs(lowerPath) do
  local curCategoryDir = lowerPath .. dir .. '/'
  config.categories[curCategory] = dir
  for file in paths.iterfiles(curCategoryDir) do
      singleImgTensor = image.load(curCategoryDir .. file)
      table.insert(imgStore, singleImgTensor)
      table.insert(lablesStore, curCategory)
  end
  curCategory = curCategory + 1
end

print(config.categories)

local trsize = #config.categories
local img = torch.Tensor(#imgStore, channels, size.y, size.x)
for i = 1, #imgStore do
  img[i] = imgStore[i]
end
local labels = torch.Tensor(lablesStore)
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
