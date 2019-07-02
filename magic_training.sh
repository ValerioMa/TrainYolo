#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo $DIR/darknet 
cd $DIR/darknet
echo  "======= Starting the training ======="
# train
#./darknet detector train ../data/cones.data ../data/yolov3-tiny-cones.cfg ../data/yolov3-tiny.conv.15 2>&1 -gpus 0 | tee ../data/backup/TrainLOG.txt
./darknet detector train ../data/cones.data $1 ../data/yolov3-tiny.conv.15 -gpus 0
