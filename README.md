# Speech Enhancement

Try to replicate Speech Denoising with Deep Feature Losses ([arXiv](https://arxiv.org/abs/1806.10522), [sound examples](https://ccrma.stanford.edu/~francois/SpeechDenoisingWithDeepFeatureLosses/))

## TODO

- look into Double-DIP algorithm

## Ideas

- If this architecture is inspired by VGG, then perhaps resnet is a better, more modern architecture (https://arxiv.org/pdf/1512.03385.pdf)
- use human voice domain dataset for acoustic scene classifier

### Getting started

```bash
sudo apt-get install sox
./scripts/setup-python.sh
./scripts/setup-gpu.sh
. env/bin/activate
python3 run.py
```

### Datasets

The Univeristy of Edinburgh [Noisy speech database](https://datashare.is.ed.ac.uk/handle/10283/2791)

The TUT Acoustic scenes 2016 [dataset](https://zenodo.org/record/45739) is used to train the scene classifier network, which is used for the loss function.

The CHiME-Home (Computational Hearing in Multisource Environments) [dataset](https://archive.org/details/chime-home) (2015)

### Data format

At the moment, this algorithm requires using 32-bit floating-point audio files to perform correctly. You can use sox to convert your file. To convert `audiofile.wav` to 32-bit floating-point audio at 16kHz sampling rate, run:

```bash
sox audiofile.wav -r 16000 -b 32 -e float audiofile.float.wav
```
