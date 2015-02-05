#!/usr/bin/env sh
~/documents/caffe/build/tools/caffe train \
    --solver=./solver_lb25.prototxt \
    --snapshot=./models/lb20_iter_25000.solverstate
