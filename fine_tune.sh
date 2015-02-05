#!/usr/bin/env sh
~/documents/caffe/build/tools/caffe train \
    --solver=./solver_fine4.prototxt \
    --weights=./models/caffenet_train_fine2_iter_10000.caffemodel \
    --gpu 0
