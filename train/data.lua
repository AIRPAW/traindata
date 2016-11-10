
--package.path = package.path .. "~/torch/install/lib/luarocks/rocks"
--package.cpath = package.cpath .. "~/torch/install/lib/luarocks/rocks"
require 'torch'
require 'image'

-- local traintDir = "/home/ira/working/images/"
traintDir = "/home/uml/working/traindata/images/"
local trsize = 3
num = 10
size = {x = 200, y = 30}
category = {"button", "checkbox", "other"}
channels = 1
local img = torch.Tensor(num*trsize,channels,size.y,size.x)
local labels = torch.Tensor(num*trsize)
local trainPortion = 0.8

for i = 1,#category do
  local index = (i-1)*num
  local name = traintDir .. category[i] .. "/"
  for j = 1, num do
    img[index+j] = image.load(name ..category[i] ..j..".jpg")
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
