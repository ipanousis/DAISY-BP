#!/usr/bin/env python

import math
import numpy

#   0	   0	
#   0	   5	
#3.54	3.54	
#   5	3.06e-16	
#3.54	-3.54	
#6.12e-16	  -5	
#-3.54	-3.54	
#  -5	-9.18e-16	
#-3.54	3.54	
#   0	  10	
#7.07	7.07	
#  10	6.12e-16	
#7.07	-7.07	
#1.22e-15	 -10	
#-7.07	-7.07	
# -10	-1.84e-15	
#-7.07	7.07	
#   0	  15	
#10.6	10.6	
#  15	9.18e-16	
#10.6	-10.6	
#1.84e-15	 -15	
#-10.6	-10.6	
# -15	-2.76e-15	
#-10.6	10.6	



daisyGrid = [[0,0]]

for rangeNo,rangeVal in enumerate([5,10,15]):
  for petalNo in range(8):
    yoff = round(math.sin((math.pi * petalNo) / 4) * rangeVal,2)
    xoff = round(math.cos((math.pi * petalNo) / 4) * rangeVal,2)
    daisyGrid.append([yoff,xoff])

#print daisyGrid

daisyGrid = numpy.int8(numpy.round(daisyGrid)).tolist()

print daisyGrid

zarr = numpy.zeros((65,65))

dataStart = 16
dataEnd   = 47
dataWidth = dataEnd-dataStart+1

outarr = numpy.ndarray(shape=(65,65),dtype=list)
sigmas = ['p%ds1','p%ds2','p%ds3']
for y in range(65):
  for x in range(65):
    content = []
    for pointNo,pointOffset in enumerate(daisyGrid):
      if(pointNo / 8 < 2): continue
      pY = y + pointOffset[0]
      pX = x + pointOffset[1]
      if(dataStart <= pY <= dataEnd and dataStart <= pX <= dataEnd):
        p = sigmas[min(pointNo / 8,2)] % ((pY-dataStart) * dataWidth + (pX-dataStart))
        content.append([pointNo,p])
    newContent = [] + content
    for data in content:
      maxx = [abs(data[0]-d[0]) for d in content if(abs(data[0]-d[0]) > 1)]
      if(len(maxx)>0): content.remove(data)

    content.sort()
    outarr[y,x] = len(content)


for row in range(65):
  print row
  print outarr[row]

print outarr.flatten().tolist().count(1),'single petal points'
print outarr.flatten().tolist().count(2),'pairs'
print outarr.flatten().tolist().count(3),'triples'
print outarr.flatten().tolist().count(4),'quadruples'
