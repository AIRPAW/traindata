classifDataLoader = {}
require 'torch'
require 'image'
require 'paths'

classifDataLoader.classifLoader = function(config)
  local traintDir = config.pathToImages
  local size = config.imagesSize
  local categories = config.categories
  local channels = config.channels
  local trainPortion = config.trainPortion
  local imgStore = {}
  local lablesStore = {}
  local singleImgTensor = torch.Tensor(channels,size.y,size.x)

  local curCategory = 1
  for dir in paths.iterdirs(traintDir) do
    local curCategoryDir = traintDir .. dir .. '/'
    config.categories[curCategory] = dir
    for file in paths.iterfiles(curCategoryDir) do
        singleImgTensor = image.load(curCategoryDir .. file)
        table.insert(imgStore, singleImgTensor)
        table.insert(lablesStore, curCategory)
    end
    curCategory = curCategory + 1
  end

  local trsize = #config.categories
  local labels = torch.Tensor(lablesStore)
  local toMix = torch.randperm(labels:size()[1])
  local trainSize = math.floor(toMix:size()[1]*trainPortion)
  local testSize = toMix:size()[1] - trainSize

  local trainData = {
    img = torch.Tensor(trainSize, channels, size.y,size.x),
    labels = torch.Tensor(trainSize),
    size = function() return trainSize end
  }

  local testData = {
    img = torch.Tensor(testSize, channels, size.y,size.x),
    labels = torch.Tensor(testSize),
    size = function() return testSize end
  }
  
  for i = 1, trainSize do
    trainData.img[i] = imgStore[toMix[i]]:clone()
    trainData.labels[i] = lablesStore[toMix[i]]
  end

  for i = 1, testSize do
    testData.img[i] = imgStore[toMix[i + trainSize]]:clone()
    testData.labels[i] = lablesStore[toMix[i + trainSize]]
  end

  collectgarbage()

  return {
    trainData,
    testData,
    trsize
  }
end

return classifDataLoader
