#!/bin/bash

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

# echo -e "\nResampling downloaded data...\n"

# mkdir -p dataset/dat
# pushd chime_home
# for d in */; do
#     pushd "$d"
#     for f in *.16kHz.wav; do
#         ff=$(echo $f | cut -d '.' -f 1-2,4) 
#         sox "$f" -e float -b 32 "../../dataset/dat/$ff" #ALREADY AT 16KHZ
#     done
#     popd
# done
# cp "development_chunks_refined.csv" "../dataset/dat/development_chunks_refined.csv"
# cp "evaluation_chunks_refined.csv" "../dataset/dat/evaluation_chunks_refined.csv"
# popd

# # REMOVE TMP DATA
# rm -r asc_tmp
# rm -r chime_home


