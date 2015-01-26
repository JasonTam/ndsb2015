#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

~/documents/caffe/build/tools/compute_image_mean ~/documents/ndsb2015/data/ndsb_train_lmdb \
  ~/documents/ndsb2015/ndsb_mean.binaryproto

echo "Done."
