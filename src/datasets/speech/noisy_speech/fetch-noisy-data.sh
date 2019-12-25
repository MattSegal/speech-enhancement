#!/bin/bash
set -e

echo -e "\nDownloading datasets...\n"
mkdir -p data/compressed
pushd data/compressed
FILES="clean_trainset_28spk_wav.zip noisy_trainset_28spk_wav.zip clean_testset_wav.zip noisy_testset_wav.zip"
for file in $FILES; do
    if [ ! -f "$file" ]; then
        echo -e "\nDownloading ${file}..."
        wget "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/${file}"
    fi
done
popd

# Clean up inflated data
echo -e "\nCleaning up old inflated data...\n"
rm -rf data/inflated

echo -e "\nInflating compressed data...\n"
mkdir -p data/inflated
pushd data/inflated
echo "Inflating training_set_clean..."
unzip -q -j ../compressed/clean_trainset_28spk_wav.zip -d training_set_clean
echo "Inflating training_set_noisy..."
unzip -q -j ../compressed/noisy_trainset_28spk_wav.zip -d training_set_noisy
echo "Inflating validation_set_clean..."
unzip -q -j ../compressed/clean_testset_wav.zip -d validation_set_clean
echo "Inflating validation_set_noisy..."
unzip -q -j ../compressed/noisy_testset_wav.zip -d validation_set_noisy
popd

echo -e "\nResampling downloaded data...\n"
DATASETS="training_set_clean training_set_noisy validation_set_clean validation_set_noisy"
# SAMPLING_RATE=16000
SAMPLING_RATE=22050
pushd data/
for dataset in $DATASETS; do
    echo "Processing $dataset ..."
    mkdir -p "$dataset"
    pushd "inflated/$dataset"
    for file in *.wav; do
        sox "$file" -e float -b 32 "../../${dataset}/${file}" rate -v -I $SAMPLING_RATE
    done
    popd
done
popd


echo -e "\nDone\n"
