#!/usr/bin/env sh

MODEL=/media/raid_arr/data/ndsb/models/pl_iter_56000.caffemodel
NET=/media/raid_arr/data/ndsb/config/train_val_pl_extract.prototxt
LAYER=fc2
OUTPUT=/media/raid_arr/data/ndsb/features/pl_56000_feats
N_MBATCH=1000
DB_TYPE=lmdb

~/documents/caffe/build/tools/extract_features.bin \
$MODEL $NET $LAYER $OUTPUT $N_MBATCH $DB_TYPE GPU DEVICE_ID=0
