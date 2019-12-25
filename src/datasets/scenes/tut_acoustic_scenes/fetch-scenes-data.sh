#!/bin/bash

echo -e "\nDownloading datasets...\n"
mkdir -p data/compressed
pushd data/compressed

# Training acoustic scenes dataset.
for i in `seq 1 8`; 
do
    if [ ! -f TUT-acoustic-scenes-2016-development.audio.$i.zip ]; then
        wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.$i.zip
    fi
done 
if [ ! -f TUT-acoustic-scenes-2016-development.meta.zip ]; then
    wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.meta.zip
fi

# Validation acoustic scenes dataset.
for i in `seq 1 3`; 
do
    if [ ! -f TUT-acoustic-scenes-2016-evaluation.audio.$i.zip ]; then
        wget https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.$i.zip
    fi
done 
if [ ! -f TUT-acoustic-scenes-2016-evaluation.meta.zip ]; then
    wget https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.meta.zip
fi

popd


# Clean up inflated data
echo -e "\nCleaning up old inflated data...\n"
rm -rf data/inflated

echo -e "\nInflating compressed data...\n"
mkdir -p data/inflated
pushd data/inflated

# Training acoustic scenes dataset.
for i in `seq 1 8`;
do
    unzip -q -j ../compressed/TUT-acoustic-scenes-2016-development.audio.$i.zip -d scenes_training_set
done
unzip -q -p ../compressed/TUT-acoustic-scenes-2016-development.meta.zip TUT-acoustic-scenes-2016-development/meta.txt > scenes_training_set/meta.txt

# Validation acoustic scenes dataset.
for i in `seq 1 3`;
do
    unzip -q -j ../compressed/TUT-acoustic-scenes-2016-evaluation.audio.$i.zip -d scenes_validation_set
done
unzip -q -p ../compressed/TUT-acoustic-scenes-2016-evaluation.meta.zip TUT-acoustic-scenes-2016-evaluation/meta.txt > scenes_validation_set/meta.txt

popd

echo -e "\nResampling downloaded data...\n"
DATASETS="scenes_training_set scenes_validation_set"
SAMPLING_RATE=16000
# SAMPLING_RATE=22050
pushd data/
for dataset in $DATASETS; do
    echo "Processing $dataset ..."
    mkdir -p "$dataset"
    pushd "inflated/$dataset"
    for file in *.wav; do
        sox "$file" -e float -b 32 "../../${dataset}/${file}" rate -v -I $SAMPLING_RATE
    done
    cp "meta.txt" "../../${dataset}/meta.txt"
    popd
done
popd
