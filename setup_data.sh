#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#sudo apt-get update && apt-get install -y \
#  openssl
  
cd $DIR

echo "======= Download openssl ======="
echo "DOWNLOAD COMPATIBLE OPENSSL DISABLED!!"
#sudo apt-get install make
#curl https://www.openssl.org/source/openssl-1.0.2l.tar.gz | tar xz && cd openssl-1.0.2l && sudo ./#config && sudo make && sudo make install
#sudo ln -sf /usr/local/ssl/bin/openssl /usr/bin/openssl
echo "======= DONE ======="

echo "======= Download dataset ======="
#curl https://transfer.sh/Ksk0o/test.dat -o data.dat
#openssl enc -aes-256-cbc -d -pass file:./key.txt < data.dat > data.tar.gz
curl https://syncandshare.lrz.de/dl/fi9FhSNEkrDAFMg6DHCZqwyX -o data.tar.gz
#rm data.dat
tar xzf ./data.tar.gz 
rm data.tar.gz
echo "======= DONE ======="

# DATASET SET UP
echo "======= Init labels folder ======="
rm $DIR/data/labels
echo ln -s $DIR/data/images $DIR/data/labels
ln -s $DIR/data/images $DIR/data/labels
echo "======= DONE ======="

echo "======= Creating bakup folder ======="
echo mkdir -p $DIR/data/backup/
mkdir -p $DIR/data/backup/
echo "======= DONE ======="

echo "======= Creating Training and Validating set ======="
cd ./data

FILE=./images/imgs_list.txt # file that store all the image list
img_n=$(wc -l < "$FILE")
valid_n=$((0.2*$img_n))  # NUMBER OF IMAGE USED FOR VALIDATION
train_n=$(($img_n-$valid_n))

echo "Total images: " $img_n
echo "Training images: " $train_n
echo "Validation images: " $valid_n

# Shuffle the lane of the file and select training and validation
sort -R $FILE > tmp_shuff.txt
echo "Creating train.txt"
head tmp_shuff.txt -n $train_n > train.txt
echo "Creating validation.txt"
tail tmp_shuff.txt -n $valid_n > validation.txt
rm tmp_shuff.txt
echo "======= DONE ======="

cd $DIR
