-- dataSetLoader предоставляет интерфейс загрузчика датасета и его подготовки
-- setLoader - фугкция, устанавливающая пользовательский загрузчик
-- loadData - исполняет функцию загрузки, передавая ей в качестве параметра
-- необязательную таблицу конфигурации, исполняемая функция должна возвращать
-- таблицу, содержащую датасет

dataSetLoader={}
require 'torch'
require 'image'
local config = require 'config'

dataSetLoader.loader = nil
dataSetLoader.setLoader = function(this, f)
    this.loader = f
end

dataSetLoader.loadData = function(this, loaderConf)
  return this.loader(loaderConf)
end

return dataSetLoader
