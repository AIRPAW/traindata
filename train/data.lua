require 'torch'
require 'image'

local traintDir = "/home/ira/working/images/"
local trsize = 3
local num = 10
local size = {x = 200, y = 30}
local category = {"button", "checkbox", "other"}
local channels = 3
local img = torch.Tensor(num*trsize,channels,size.x,size.y)
local labels = torch.Tensor(num*trsize)

for i = 1,#category do
  local index = (i-1)*num
  local name = traintDir .. category[i] .. "/"
  for j = 1, num do
    img[index+j] = image.load(name ..category[i] ..j..".jpg")
    labels[index+j] = i
    print(category[i])
  end
end
