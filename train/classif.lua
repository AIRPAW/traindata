
-- нужно запустить с именем картинки из папки test_path
require 'torch'
require 'optim'
require 'xlua'
require 'image'
require 'nn'
require 'nngraph'

model_file = '/home/sbt-voronova-id/traindata/models/models'
test_path = '/home/sbt-voronova-id/traindata/test_img/'

torch.setdefaulttensortype('torch.DoubleTensor')
local config = require 'config'
local channels = config.channels
local size = config.size
local categories = config.categories
local name_img = arg[1]

local config = {
  momentum          = 0,
  learningRate      = 1e-2,
  weightDecay       = 1e-5,
  learningRateDecay = 1e-7,
}

local m = torch.load(model_file)

local input = image.load(test_path .. name_img)
local inp = torch.Tensor(input)
local predicted = m:forward(inp)

image.display(input)
print("predicted: ")
print(predicted)
local p = torch.exp(predicted)
local mx, max_i = torch.max(p, 1)

for i = 1, predicted:size(1) do
  if (max_i[1] == i) then
    print(sys.COLORS.green .. categories[i], torch.exp(predicted[i]))
  else
    print(sys.COLORS.white .. categories[i], torch.exp(predicted[i]))
  end
  p = p + torch.exp(predicted[i])
end
