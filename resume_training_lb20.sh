#!/usr/bin/env sh
~/documents/caffe/build/tools/caffe train \
    --solver=./solver_lb20.prototxt \
    --snapshot=./models/lb15_iter_20000.solverstate
