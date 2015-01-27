#!/usr/bin/env sh
cp ./train_val.prototxt ./models
cp ./solver.prototxt ./models
~/documents/caffe/build/tools/caffe train \
    --solver=/afs/ee.cooper.edu/user/t/a/tam8/documents/ndsb2015/solver.prototxt
