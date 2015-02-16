#!/usr/bin/env sh
cp ./train_val.prototxt ./models
cp ./solver.prototxt ./models
~/documents/caffe/build/tools/caffe train \
    --solver=./solver.prototxt \
    --snapshot=./models/caffenet_train_iter_5000.solverstate
