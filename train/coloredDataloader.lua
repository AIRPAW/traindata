coloredDataloader = {}
require 'torch'
require 'image'
require 'paths'

coloredDataloader.coloerdLoader = function(config)
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
    local singleMarkTensor = torch.Tensor(1,size.y,size.x)

    for file in paths.iterfiles(trainDir) do
      singleImgTensor = image.load(trainDir .. file)
      singleMarkTensor = image.load(trainDir .. "codes/" .. file)
      table.insert(imgStore, singleImgTensor)
      table.insert(markedStore, singleMarkTensor)
    end

    local toMix = torch.randperm(#imgStore)
    local trainSize = math.floor(toMix:size()[1]*trainPortion)
    local testSize = toMix:size()[1] - trainSize
    local trainData = {
      img = torch.Tensor(trainSize, channels, size.y,size.x),
      marks = torch.Tensor(trainSize, 1, size.y,size.x),
      size = function() return trainSize end
    }

    local testData = {
      img = torch.Tensor(testSize, channels, size.y,size.x),
      marks = torch.Tensor(testSize, 1, size.y,size.x),
      size = function() return testSize end
    }

    for i = 1, trainSize do
print(trainData.img[i]:size())
print(imgStore[toMix[i]]:size())
      trainData.img[i] = imgStore[toMix[i]]:clone()
      trainData.marks[i] = markedStore[toMix[i]]:clone()
    end

    for i = 1, testSize do
      testData.img[i] = imgStore[toMix[i + trainSize]]:clone()
      testData.marks[i] = markedStore[toMix[i + trainSize]]:clone()
    end

    return {
      trsize = trsize,
      trainData = trainData,
      testData = testData
    }
  end
end

return coloredDataloader
