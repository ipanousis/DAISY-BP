#!/usr/bin/env python

import math
import numpy

from matplotlib import pyplot as plt

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
#  10	6.12e-1	
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

dataWidthX, dataWidthY = 16,16

size = max(dataWidthX*3,dataWidthY*3)
zarr = numpy.zeros((size,size))


tripleListXYR = [] # x,y position and ratio of outputs over window data

if(0):
  for dataWidthX in range(8,34,2):

    dataStartX = size/2-dataWidthX/2
    dataEndX = dataStartX+dataWidthX-1

    print 'from',dataStartX

    for dataWidthY in range(8,34,2):

      dataStartY = size/2-dataWidthY/2
      dataEndY = dataStartY+dataWidthY-1


outarr = numpy.ndarray(shape=zarr.shape,dtype=list)
sigmas = ['p%ds1','p%ds2','p%ds3']
#for dataStartY in range(0,128,dataWidth):
#  dataEndY = dataStartY + dataWidth
#  for dataStartX in range(0,128,dataWidth):
#    dataEndX = dataStartX + dataWidth
for dataStartX in range(0,size,4):
  dataEndX = dataStartX+dataWidthX-1
  for dataStartY in range(0,size,4):
    dataEndY = dataStartY+dataWidthY-1
    for y in range(size):
      for x in range(size):
        newContent = []
        for pointNo,pointOffset in enumerate(daisyGrid):
          if((pointNo-1) / 8 != 2): continue
          pY = y + pointOffset[0]
          pX = x + pointOffset[1]
          if(dataStartY <= pY <= dataEndY and dataStartX <= pX <= dataEndX):
            newContent.append((pointNo-1) % 8)
        content = [] + newContent
        for data in newContent:
          diff = [abs(data-d) for d in content]
          if(len(diff)>0 and diff.count(1)==0): content.remove(data)

        if(outarr[y,x] == None):
          outarr[y,x] = []
        for item in content:
          if(not (item in outarr[y,x])):
            outarr[y,x].append(item)
        outarr[y,x].sort()
        #outarr[y,x] = content

if(0):
  tripleListXYR.append([dataWidthX,dataWidthY,(outarr.flatten().tolist().count(2)*2.0+\
                                               outarr.flatten().tolist().count(3)*2.0+\
                                               outarr.flatten().tolist().count(4)*4.0+\
                                               outarr.flatten().tolist().count(5)*4.0) / (dataWidthX*dataWidthY)])

  widthsX = [item[0] for item in tripleListXYR]
  widthsY = [item[1] for item in tripleListXYR]
  ratios  = [item[2] for item in tripleListXYR]

  plt.scatter(widthsX, widthsY, c=ratios)
  plt.colorbar()
  plt.show()

#for row in range(size):
#  print row
#  print outarr[row]

outarrlist = outarr[dataWidthY:size-dataWidthY,dataWidthX:size-dataWidthX].flatten().tolist()
for total in range(1,9):
  print map(len,outarrlist).count(total),'%d\'s'%total

