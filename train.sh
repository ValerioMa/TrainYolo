#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo $DIR/darknet 
cd $DIR/darknet
echo  "======= Starting the training ======="
# train
let count = 0
for doc in ../data/param/*
do
	((count++))	
	touch ../data/backup/TrainLOG_$count.txt	
	echo $doc "in Training!"	
	./darknet detector train ../data/cones.data $doc
	../data/yolov3-tiny.conv.15
done
