#!/bin/bash
set -e

echo -e "\nDownloading datasets...\n"
mkdir -p data/compressed
pushd data/compressed
if [ ! -f chime_home.tar.gz ]; then
    wget https://archive.org/download/chime-home/chime_home.tar.gz
fi
popd


# Clean up inflated data
echo -e "\nCleaning up old inflated data...\n"
rm -rf data/inflated

echo -e "\nInflating compressed data...\n"
mkdir -p data/inflated
pushd data/inflated
tar -xzf ../compressed/chime_home.tar.gz -C ./
popd

echo -e "\nResampling downloaded data...\n"
mkdir -p data/chime/audio
mkdir -p data/chime/labels
pushd data/inflated/chime_home/chunks
# Copy file metadata
for file in *.csv; do
    cp "$file" "../../../chime/labels/$file"
done
# Resample audio files
for file in *.16kHz.wav; do
    new_filename=$(echo $file | cut -d '.' -f 1-2,4) 
    # Already sampled at 16kHz
    sox "$file" -e float -b 32 "../../../chime/audio/$new_filename"
done
popd

pushd data
cp "inflated/chime_home/development_chunks_refined.csv" "chime/development_chunks_refined.csv"
cp "inflated/chime_home/evaluation_chunks_refined.csv" "chime/evaluation_chunks_refined.csv"
popd
