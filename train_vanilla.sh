#!/usr/bin/env sh
cp ./train_val_vanilla.prototxt ./models
cp ./solver_vanilla.prototxt ./models
~/documents/caffe/build/tools/caffe train \
    --solver=/afs/ee.cooper.edu/user/t/a/tam8/documents/ndsb2015/solver_vanilla.prototxt
