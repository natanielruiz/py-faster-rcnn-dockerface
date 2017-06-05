#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

TRAIN_IMDB_ONE="coco_reduced_2014_train"
TRAIN_IMDB_TWO="kitchen_imagenet_train"
PT_DIR="kitchen"
ITERS=160000

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net_doubledb.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb_one ${TRAIN_IMDB_ONE} \
  --imdb_two ${TRAIN_IMDB_TWO} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

# time ./tools/test_net.py --gpu ${GPU_ID} \
#   --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
#   --net ${NET_FINAL} \
#   --imdb ${TEST_IMDB} \
#   --cfg experiments/cfgs/faster_rcnn_end2end.yml \
#   ${EXTRA_ARGS}
