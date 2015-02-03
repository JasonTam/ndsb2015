#!/usr/bin/env sh
~/documents/caffe/build/tools/caffe train \
    --solver=./solver_fine1.prototxt \
    --weights=./models/caffenet_train_iter_25000.caffemodel \
    --gpu 0
