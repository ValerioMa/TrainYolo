#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo $DIR/darknet 
cd $DIR/darknet
echo  "======= Starting the training ======="
touch ../data/backup/TrainLOG.txt
# train
./darknet detector train ../data/cones.data ../data/yolov3-tiny-cones.cfg ../data/yolov3-tiny.conv.15 &> ../data/backup/TrainLOG.txt
