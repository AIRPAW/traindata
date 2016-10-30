
package.path = "/home/uml/.luarocks/share/lua/5.1/?.lua;/home/uml/.luarocks/share/lua/5.1/?/init.lua;/home/uml/torch/install/share/lua/5.1/?.lua;/home/uml/torch/install/share/lua/5.1/?/init.lua;./?.lua;/home/uml/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua;/home/uml/torch/install/share/lua/5.2/?.lua;/home/uml/torch/install/share/lua/5.2/?/init.lua"
package.cpath = "/home/uml/.luarocks/lib/lua/5.1/?.so;/home/uml/torch/install/lib/lua/5.1/?.so;/home/uml/torch/install/lib/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so;/home/uml/torch/install/lib/lua/5.2/?.so	"
require 'torch'
require 'image'

-- local traintDir = "/home/ira/working/images/"
local traintDir = "/home/uml/working/traindata/images/"
local trsize = 3
local num = 10
local size = {x = 200, y = 30}
local category = {"button", "checkbox", "other"}
local channels = 1
local img = torch.Tensor(num*trsize,channels,size.x,size.y)
local labels = torch.Tensor(num*trsize)
local trainPortion = 0.9

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
  img = torch.Tensor(trainSize, channels, size.x, size.y),
  labels = torch.Tensor(trainSize),
  size = function() return trainSize end
}

testData = {
  img = torch.Tensor(testSize, channels, size.x, size.y),
  labels = torch.Tensor(testSize),
  size = function() return testSize end
}

for i = 1, trainSize do
  trainData.img[i] = img[toMix[i]]:clone()
  trainData.labels[i] = labels[toMix[i]]
  --image.display(trainData.img[i][1])
end

for i = 1, testSize do
  trainData.img[i] = img[toMix[i + trainSize]]:clone()
  trainData.labels[i] = labels[toMix[i + trainSize]]
end
