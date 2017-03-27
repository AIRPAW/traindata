
-- нужно запустить с именем картинки из папки test_path
require 'torch'
require 'optim'
require 'xlua'
require 'image'
require 'nn'
require 'nngraph'

local config = require 'config'

local model_file = config.modelPath .. 'model'
local test_path = config.pathToTestImages

torch.setdefaulttensortype('torch.FloatTensor')
local channels = config.channels
local size = config.size
local categories = config.categories
local name_img = arg[1]

local m = torch.load(model_file)
local input = image.load(test_path .. name_img)
local inp = torch.Tensor(input)
local predicted =  torch.Tensor(config.channels, config.imagesSize.x, config.imagesSize.y)
predicted = m:forward(inp)
image.display(predicted)
