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
#curl https://syncandshare.lrz.de/dl/fi9FhSNEkrDAFMg6DHCZqwyX -o data.tar.gz # augmented data
curl https://syncandshare.lrz.de/dl/fiUKmjiaSAtJytRLij7mMHQe -o data.tar.gz # to be augmented
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

echo "======= Creating Validating set ======="
cd ./data

FILE=../data/images/imgs_list.txt # file that store all the image list
touch $FILE
img_n=$(wc -l < "$FILE")
valid_n=$(($img_n/5))  # NUMBER OF IMAGE USED FOR VALIDATION

# Shuffle the lane of the file and select training and validation
touch tmp_shuff.txt
touch validation.txt
sort -R $FILE > tmp_shuff.txt
echo "Creating validation.txt"
tail -n $valid_n tmp_shuff.txt > validation.txt
rm tmp_shuff.txt

echo "======= Creating Python Environment ======="

export PYTHONPATH=$PYTHONPATH:$(pwd)/../scripts
python ../scripts/image-label-converter.py
python ../scripts/data-augmentation.py
python ../scripts/image-label-converter_txt.py

echo "======= Creating Training and Validating set ======="

echo "Creating train.txt"

touch tmp_shuff.txt
touch train.txt
ls ../data/images/*.jpg > tmp_shuff.txt
grep -Fvxf validation.txt tmp_shuff.txt > train.txt
FILE2=../data/validation.txt # file that store all the new images
touch $FILE2
img_new=$(wc -l < "$FILE")
rm tmp_shuff.txt

train_n=$(($img_new-$valid_n))

echo "Total images: " $img_new
echo "Training images: " $train_n
echo "Validation images: " $valid_n

echo "======= DONE ======="

cd $DIR
