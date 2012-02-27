#!/usr/bin/env python

import numpy
import Image
import os

gpufile = open('sDSC01973.jpg.bdaisy.gpu','rb')
cpufile = open('sDSC01973.jpg.cpu_unnorm','rb')

os.lseek(gpufile.fileno(), 4 * 4, os.SEEK_SET)
os.lseek(cpufile.fileno(), 4 * 4, os.SEEK_SET)

imageWidth = 1024
readFromElementYX = 256 * imageWidth + 256
jumpElements = readFromElementYX * 200


os.lseek(gpufile.fileno(), jumpElements * 4, os.SEEK_SET)
os.lseek(cpufile.fileno(), jumpElements * 4, os.SEEK_SET)

gpuarr = numpy.fromfile(gpufile, dtype=numpy.float32, count=200)
cpuarr = numpy.fromfile(cpufile, dtype=numpy.float32, count=200)

print 'cpu',cpuarr*255
print 'gpu',gpuarr

print 'Element at %d' % readFromElementYX
