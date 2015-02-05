#!/usr/bin/env sh
~/documents/caffe/build/tools/caffe train \
    --solver=./solver_lb30.prototxt \
    --snapshot=./models/lb25_iter_30000.solverstate
