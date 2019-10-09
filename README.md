# Speech Enhancement

Try to replicate Speech Denoising with Deep Feature Losses ([arXiv](https://arxiv.org/abs/1806.10522), [sound examples](https://ccrma.stanford.edu/~francois/SpeechDenoisingWithDeepFeatureLosses/))

Code and concepts borrowed from [Francois Germain's repo](https://github.com/francoisgermain/SpeechDenoisingWithDeepFeatureLosses)

**Status**: Currently trying to train the audio classifier network that is used for the loss function - training goes poorly.

## RUNS

- A
  small dataset (18 batches of 28), 1 feature layer, track MSE, see what 10 epochs achieves
  what does small dataset do?
  how well does MSE loss correspond to good audio output?
  is 1 top of stack feature vector enough for the loss function?
  then check model quality in notebook

  in general loss and MSE are both decreasing (which is good)
  both training and validation loss / MSE are very similar (yay!)
  improvement plateus at about about 12 epochs
  ~0.95 loss
  ~0.009 MSE

- B
  same as A but whole dataset (414 batches of 28)

## TODO

- Look into raw audio tips
- Try train a resnet / VGG on images to sanity check architecture
- Try to reproduce an image denoising algo

## Ideas

- use human voice domain dataset for acoustic scene classifier
- look into Double-DIP algorithm for further ideas on denoising

### Getting started

This should get you 80% of the way there.

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
http://www.cs.tut.fi/~mesaros/pubs/mesaros_eusipco2016-dcase.pdf

The CHiME-Home (Computational Hearing in Multisource Environments) [dataset](https://archive.org/details/chime-home) (2015)

### Data format

At the moment, this algorithm requires using 32-bit floating-point audio files to perform correctly. You can use sox to convert your file. To convert `audiofile.wav` to 32-bit floating-point audio at 16kHz sampling rate, run:

```bash
sox audiofile.wav -r 16000 -b 32 -e float audiofile.float.wav
```
