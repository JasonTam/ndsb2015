#!/usr/bin/env sh
# Create the ndsb lmdb inputs
# N.B. set the path to the ndsb train + test data dirs
# --gray flag is used in call to treat images as grayscales

EXAMPLE=~/documents/ndsb2015/data
DATA=~/documents/ndsb2015/data
TRAINTXT=train0.txt
TESTTXT=test0.txt
TOOLS=~/documents/caffe/build/tools

TRAIN_DATA_ROOT=/.	# not important cuz I use abs path
TEST_DATA_ROOT=/.       # not important cuz I use abs path

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=64
  RESIZE_WIDTH=64
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_ndsb.sh to the path" \
       "where the ndsb training data is stored."
  exit 1
fi

if [ ! -d "$TEST_DATA_ROOT" ]; then
  echo "Error: TEST_DATA_ROOT is not a path to a directory: $TEST_DATA_ROOT"
  echo "Set the TEST_DATA_ROOT variable in create_ndsb.sh to the path" \
       "where the ndsb TEST data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --gray \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/$TRAINTXT \
    $EXAMPLE/ndsb_train_lmdb

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --gray \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TEST_DATA_ROOT \
    $DATA/$TESTTXT \
    $EXAMPLE/ndsb_test_lmdb

echo "Done."
