localPath         = {
getLocalPath = function ()
                  return '/home/ml/traindata/'
               end
}

local config = {
  localPath         = localPath.getLocalPath(),
  batchSize         = 4,
  momentum          = 0,
  learningRate      = 1e-2,
  weightDecay       = 1e-3,
  learningRateDecay = 1e-7,
  epochnm           = 500,
  modelPath         = localPath.getLocalPath() .. 'models/',
  with_plotting     = false,
  data_file_path    = localPath.getLocalPath() .. 'data/save.dat',
  pathToImages      = localPath.getLocalPath() .. 'Dataset/',
  pathToTestImages  = localPath.getLocalPath() .. 'Dataset/',
  categories        = {},
  imagesSize        = {x = 400, y = 240},
  channels          = 1,
  trainPortion      = 0.7,
}



return config
