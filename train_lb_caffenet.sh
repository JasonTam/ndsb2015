#!/usr/bin/env sh
cp ./train_val_lb.prototxt ./models
cp ./solver_lb.prototxt ./models
~/documents/caffe/build/tools/caffe train \
    --solver=/afs/ee.cooper.edu/user/t/a/tam8/documents/ndsb2015/solver_lb.prototxt
