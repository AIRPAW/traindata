localPath         = {
getLocalPath = function ()
                  return '/home/ml/train'
               end
}

local config = {
  localPath         = localPath.getLocalPath(),
  batchSize         = 3,
  momentum          = 0,
  learningRate      = 1e-2,
  weightDecay       = 1e-3,
  learningRateDecay = 1e-7,
  epochnm           = 2,
  modelPath         = localPath.getLocalPath() .. 'models/',
  with_plotting     = false,
  data_file_path    = localPath.getLocalPath() .. 'data/save.dat',
  pathToImages      = localPath.getLocalPath() .. 'Dataset/',
  pathToTestImages  = localPath.getLocalPath() .. 'test_img/',
  categories        = {},
  imagesSize        = {x = 480, y = 320},
  channels          = 3,
  trainPortion      = 0.7,
}



return config
