plotv = {}

function plotv.plotv(file)
  gnuplot.grid(true)
  local raw_str1 = string.format("plot '%s' using 1:2 title 'train valid' with  linespoints lw 1, ", file)
  local raw_str2 = string.format("'%s' using 1:3 title 'test valid' with  linespoints lw 1", file)
  gnuplot.raw((raw_str1 .. raw_str2))
  gnuplot.plotflush()
end

return plotv
