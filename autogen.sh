#!/bin/sh

touch NEWS README AUTHORS ChangeLog
autoreconf --force --install
./configure
ln -s src/daisy/daisyKernels.cl
