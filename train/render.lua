
--package.path = package.path .. "/home/uml/torch-opencv/modules"
--package.cpath = package.cpath .. "/home/uml/torch-opencv/modules"
require 'torch'
require 'image'
require 'paths'

local cv = require 'cv'
local config = require 'config'
require 'cv.imgproc'
require 'cv.imgcodecs'

local dataDir = config.localPath .. "images_/"
local pathToSave = config.localPath .. "render_images/"
local loadType = cv.IMREAD_GRAYSCALE
local size = config.imagesSize


paths.rmall(pathToSave,"yes")
paths.mkdir(pathToSave)

for dir in paths.iterdirs(dataDir) do
   print(sys.COLORS.red .. "do files from " .. dir)
   local k = 1;
   local imagePath = dataDir .. dir .. '/'
   local saveDir = pathToSave .. dir .. '/'
   paths.mkdir(pathToSave .. dir)
   for file in paths.iterfiles(imagePath) do
     im = cv.imread{imagePath .. file, loadType}
     local dst = im:clone()
     cv.adaptiveThreshold{src = im,dst = dst, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv.THRESH_BINARY_INV, blockSize=11, C=2}
     local dst_save = cv.resize{dst, {size.x, size.y}, interpolation=cv.INTER_CUBIC}
     local file_name = dir .. k .. ".jpg"
     cv.imwrite{saveDir ..file_name, dst_save}
     k = k+1
   end
end
