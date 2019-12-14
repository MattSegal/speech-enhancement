# Speech Enhancement

Tinkering with speech enhancement models.

Borrowed code, models and techniques from:

- Improved Speech Enhancement with the Wave-U-Net (([arXiv](https://arxiv.org/abs/1811.11307))
- Wave-U-Net: a multi-scale neural network for end-to-end audio source separation ([arXiv](https://arxiv.org/pdf/1806.03185.pdf))
- Speech Denoising with Deep Feature Losses ([arXiv](https://arxiv.org/abs/1806.10522), [sound examples](https://ccrma.stanford.edu/~francois/SpeechDenoisingWithDeepFeatureLosses/), [GitHub](https://github.com/francoisgermain/SpeechDenoisingWithDeepFeatureLosses))
- MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis ([arXiv](https://arxiv.org/abs/1910.06711), [sound examples](https://melgan-neurips.github.io/), [GitHub](https://github.com/seungwonpark/melgan))

### Datasets

The following datasets are used:

- The Univeristy of Edinburgh [Noisy speech database](https://datashare.is.ed.ac.uk/handle/10283/2791) for speech enhancement problem
- The TUT Acoustic scenes 2016 [dataset](https://zenodo.org/record/45739) is used to train the scene classifier network, which is used for the loss function. ([dataset paper](http://www.cs.tut.fi/~mesaros/pubs/mesaros_eusipco2016-dcase.pdf))
- The CHiME-Home (Computational Hearing in Multisource Environments) [dataset](https://archive.org/details/chime-home) (2015) is also used for the scene classifier, in some experiments
- The "train-clean-100" dataset from [Librispeech](http://www.openslr.org/12), mixed with the TUT acoustic scenes dataset.

### Data format

At the moment, the algorithm uses 32-bit floating-point audio files at a 16kHz sampling rate to perform correctly. You can use `sox` to convert your file. To convert `audiofile.wav` to 32-bit floating-point audio at 16kHz sampling rate, run:

```bash
sox audiofile.wav -r 16000 -b 32 -e float audiofile.float.wav
```
