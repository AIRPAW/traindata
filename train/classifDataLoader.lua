classifDataLoader = {}
require 'torch'
require 'image'
require 'paths'

classifDataLoader.classifLoader = function()

end
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



print("dataset loaded")

return {
  trainData,
  testData,
  trsize
}
