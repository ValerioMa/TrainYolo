#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR

sudo apt-get update && apt-get install -y \
  autoconf \
  automake \
  libtool \
  build-essential \
  git \
  wget

# DARKNET
echo "======= Cloning Darknet ======="
git clone https://github.com/AlexeyAB/darknet.git
echo "======= DONE ======="

echo "======= Patch Darknet ======="
#echo "GPU PATCH DISABLED"
patch ./darknet/Makefile ./patch/make_GPU.patch
echo "DARKNET PATCH DISABLED"
#patch ./darknet/examples/detector.c ./patch/backup_freq.patch 
cp MagicMakefile ./darknet/Makefile
echo "======= DONE ======="

echo "======= Compiling darknet ======="
cd ./darknet
make
echo "======= DONE ======="

cd $DIR


