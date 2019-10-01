# Speech Enhancement

Try to replicate Speech Denoising with Deep Feature Losses ([arXiv](https://arxiv.org/abs/1806.10522), [sound examples](https://ccrma.stanford.edu/~francois/SpeechDenoisingWithDeepFeatureLosses/))

TODO

- try out sacred https://sacred.readthedocs.io/en/stable/experiment.html

### Getting started

```bash
sudo apt-get install sox
./scripts/setup-gpu.sh
./scripts/setup-python.sh
jupyter notebook
```

### Data format

At the moment, this algorithm requires using 32-bit floating-point audio files to perform correctly. You can use sox to convert your file. To convert `audiofile.wav` to 32-bit floating-point audio at 16kHz sampling rate, run:

```bash
sox audiofile.wav -r 16000 -b 32 -e float audiofile.float.wav
```
