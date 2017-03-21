coloredDataloader = {}
require 'torch'
require 'image'
require 'paths'

coloredDataloader.coloerdLoader = function(this, config)
  if config == nil then
    error("Set configuration for screen loader")
  else
    local trainDir = config.pathToImages
    local size = config.imagesSize
    local categories = config.categories
    local channels = config.channels
    local trainPortion = config.trainPortion
    local imgStore = {}
    local markedStore = {}
    local singleImgTensor = torch.Tensor(channels,size.y,size.x)
    local singleMarkTensor = torch.Tensor(channels,size.y,size.x)
    for file in paths.iterfiles(trainDir) do
      singleImgTensor = image.load(trainDir .. file)
      singleMarkTensor = image.load(trainDir .. "marked/" .. file)
      table.insert(imgStore, singleImgTensor)
      table.insert(markedStore, singleMarkTensor)
    end
  end

  local toMix = torch.randperm(imgStore:size()[1])
  local trainSize = math.floor(toMix:size()[1]*trainPortion)
  local testSize = toMix:size()[1] - trainSize

  local trainData = {
    img = torch.Tensor(trainSize, channels, size.y,size.x),
    marks = torch.Tensor(trainSize),
    size = function() return trainSize end
  }

  local testData = {
    img = torch.Tensor(testSize, channels, size.y,size.x),
    marks = torch.Tensor(testSize),
    size = function() return testSize end
  }

  for i = 1, trainSize do
    trainData.img[i] = imgStore[toMix[i]]:clone()
    trainData.marks[i] = markedStore[toMix[i]]
  end

  for i = 1, testSize do
    testData.img[i] = imgStore[toMix[i + trainSize]]:clone()
    testData.marks[i] = markedStore[toMix[i + trainSize]]
  end

  collectgarbage()

  return {
    trsize,
    trainData,
    testData
  }
end

return coloredDataloader
