#!/usr/bin/env sh
cp ./train_val_lb.prototxt ./models
cp ./solver_lb.prototxt ./models
~/documents/caffe/build/tools/caffe train \
    --solver=./solver_lb10.prototxt \
    --snapshot=./models/lb_iter_10000.solverstate
