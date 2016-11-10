
-- нужно запустить с именем картинки из папки test_path
require 'torch'
require 'optim'
require 'xlua'
require 'image'
require 'nn'
require 'nngraph'

model_file = '/home/uml/working/traindata/models/models'
test_path = '/home/uml/working/traindata/test_img/'

local channels = 1
local size = {x = 200, y = 30}
local category = {"button", "checkbox", "input", "other"}
local name_img = arg[1]

local config = {
  momentum          = 0,
  learningRate      = 1e-2,
  weightDecay       = 1e-5,
  learningRateDecay = 1e-7,
}

local m = torch.load(model_file)

local input = image.load(test_path .. name_img)
local predicted = m:forward(input)
print(predicted)
