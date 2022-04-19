# Wave-U-Net-VoiceComplement-Pytorch

## Thanks

This repository was created based on [Wave-U-Net (Pytorch)](https://github.com/f90/Wave-U-Net-Pytorch).\
All of the ideas that underlie it came from Wave-U-Net.\
Thanks:)

## Quick start

```
docker build -t unet_voice_complement .
docker run --shm-size 8gb -v .:/app -it unet_voice_complement bash
python train.py
```

## About

This repository was created for the purpose of voice completion.\
Therefore, the following changes were made to Wave-U-Net.

### Dataset

Instead of downloading a dataset and training with it, we have changed it so that you can train with your own voices and background sounds.\
The default settings are to load the voice in the `voice` directory and the background sound in the `bgm` directory when saved as a `.wav` file.

### Model

With the above changes, the inference for multiple instruments is now made for a single instrument (i.e., voice).

### Type Hint

added.
