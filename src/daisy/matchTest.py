#!/usr/bin/env python

import pdb
import sys,os,numpy

descriptorLength = 200

daisy1 = sys.argv[1]
daisy2 = sys.argv[2]
proc = sys.argv[3]

imageHeight = int(sys.argv[4])
imageWidth = int(sys.argv[5])

if(len(sys.argv) > 6):
  searchWindowH = int(sys.argv[6])
  searchWindowW = int(sys.argv[7])
else:
  searchWindowH = 21
  searchWindowW = 151

dfile1 = open(daisy1, 'rb')
dfile2 = open(daisy2, 'rb')

darr1 = numpy.fromfile(dfile1, dtype=numpy.float32, count=imageHeight*imageWidth*descriptorLength+4)
darr1 = darr1[4:]

darr2 = numpy.fromfile(dfile2, dtype=numpy.float32, count=imageHeight*imageWidth*descriptorLength+4)
darr2 = darr2[4:]

matchOffsetY = numpy.zeros(shape=(imageHeight,imageWidth),dtype=int)
matchOffsetX = numpy.zeros(shape=(imageHeight,imageWidth),dtype=int)
matchDiff    = numpy.ones(shape=(imageHeight,imageWidth),dtype=float) * 9999

for pixelY in range(imageHeight):
  for pixelX in range(imageWidth):
    searchYRange = range(max(0, pixelY - searchWindowH / 2), min(imageHeight, pixelY + searchWindowH / 2))
    searchXRange = range(max(0, pixelX - searchWindowW / 2), min(imageWidth, pixelX + searchWindowW / 2))

    offset = pixelY * imageWidth * descriptorLength + pixelX * descriptorLength
    desc1 = darr1[offset:offset+descriptorLength]
    
    for otherY in searchYRange:
      for otherX in searchXRange:
        
        offset = otherY * imageWidth * descriptorLength + otherX * descriptorLength
        desc2 = darr2[offset:offset+descriptorLength]

        diff = numpy.sum(numpy.abs(desc2-desc1))
        if(diff < matchDiff[pixelY,pixelX]):
          matchDiff[pixelY,pixelX] = diff
          matchOffsetY[pixelY,pixelX] = otherY-pixelY
          matchOffsetX[pixelY,pixelX] = otherX-pixelX

  print pixelY

matchOffsetY.tofile(open(proc+'MatchYOffset.bin','wb'))
matchOffsetX.tofile(open(proc+'MatchXOffset.bin','wb'))
matchDiff.tofile(open(proc+'MatchDiff.bin','wb'))

pdb.set_trace()
