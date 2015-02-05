#!/usr/bin/env sh
cp ./train_val_lb.prototxt ./models
cp ./solver_lb.prototxt ./models
~/documents/caffe/build/tools/caffe train \
    --solver=./solver_lb15.prototxt \
    --snapshot=./models/lb10_iter_15000.solverstate
