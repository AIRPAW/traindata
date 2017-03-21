localPath         = {
getLocalPath = function ()
                  return '/home/sbt-voronova-id/traindata/'
               end
}

local config = {
  localPath         = localPath.getLocalPath(),
  batchSize         = 30,
  momentum          = 0,
  learningRate      = 1e-2,
  weightDecay       = 1e-3,
  learningRateDecay = 1e-7,
  epochnm           = 500,
  modelPath         = localPath.getLocalPath() .. 'models/',
  with_plotting     = true,
  data_file_path    = localPath.getLocalPath() .. 'data/save.dat',
  pathToImages      = localPath.getLocalPath() .. 'images/',
  pathToTestImages  = localPath.getLocalPath() .. 'test_img/',
  categories        = {},
  imagesSize        = {x = 200, y = 30},
  channels          = 1,
  trainPortion      = 0.7,
  numImages         = 10
}



return config
