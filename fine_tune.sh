#!/usr/bin/env sh
~/documents/caffe/build/tools/caffe train \
    --solver=./solver_fine2.prototxt \
    --weights=./models/caffenet_train_fine1_iter_15000.caffemodel \
    --gpu 0
