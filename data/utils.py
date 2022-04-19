from glob import glob
import math
import os
import random
import librosa
import numpy as np
import soundfile
import torch


def crop_targets(mix, targets, shapes):
    '''
    Crops target audio to the output shape required by the model given in "shapes"
    '''
    for key in targets.keys():
        if key != "mix":
            targets[key] = targets[key][:, shapes["output_start_frame"]:shapes["output_end_frame"]]
    return mix, targets


def load(path, sr=22050, mono=True, mode="numpy", offset=0.0, duration=None):
    y, curr_sr = librosa.load(
        path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

    if len(y.shape) == 1:
        # Expand channel dimension
        y = y[np.newaxis, :]

    if mode == "pytorch":
        y = torch.tensor(y)

    return y, curr_sr


def write_wav(path, audio, sr):
    soundfile.write(path, audio.T, sr, "PCM_16")


def resample(audio, orig_sr: int, new_sr: int, mode="numpy"):
    if orig_sr == new_sr:
        return audio

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    out = librosa.resample(audio, orig_sr, new_sr,  # type: ignore
                           res_type='kaiser_fast')

    if mode == "pytorch":
        out = torch.tensor(out)
    return out


def get_ratioFS(audio: torch.Tensor):
    dim1 = audio[0]
    rms = (dim1.square().sum() / dim1.size(0)).sqrt().item()
    ratio = rms / 32768.0
    return ratio


def get_overlay_ratio(src: torch.Tensor, bgm: torch.Tensor):
    # -8db = 0.3981, -9db = 0.35481, -7db = 0.446668
    target_ratio_diff = random.uniform(0.406668, 0.31481)

    src_ratioFS = get_ratioFS(src)
    bgm_ratioFS = get_ratioFS(bgm)

    current_ratio = bgm_ratioFS / src_ratioFS
    ratio = target_ratio_diff / current_ratio

    return ratio


def overlay_bgm(src: torch.Tensor, bgm: torch.Tensor, ratio: float):
    bgm_start = random.randrange(0, bgm.size(1) - src.size(1))

    scaled_bgm = bgm[:, bgm_start:bgm_start+src.size(1)] * ratio
    return src + scaled_bgm


def mask_audio(src: torch.Tensor, mode: str, relative_center: int, entire_length: int, sr: int):
    center = relative_center + int(entire_length * 0.1 * np.random.randn())
    duration = max(int(sr * 0.2 + sr * 0.1 * np.random.randn()), 0)
    start = center - duration // 2
    end = center + duration // 2
    if start < 0:
        start = 0
    if end >= src.size(1) - 1:
        end = src.size(1) - 1
    masked = src.clone()

    if mode == 'silent':
        mask = torch.ones(src.size())
        for i in range(mask.size(0)):
            for j in range(start, end):
                mask[i][j] = 0.0
        masked = masked * mask
    if mode == 'p':
        fs = random.randint(450, 1200)
        delta_per_frame = 2 * math.pi * fs / sr
        omega0 = random.uniform(0.0, 2 * math.pi)
        ratio = src[0].max().item() * 0.4
        # ratio = 1.0
        for i in range(masked.size(0)):
            for j in range(start, end):
                masked[i][j] = math.sin(omega0 + delta_per_frame * j) * ratio

    return masked


def separate_train_val_test(root_dir: str):
    train_val_test_list = glob(os.path.join(
        root_dir, '**', '*.wav'), recursive=True)
    print(len(train_val_test_list), train_val_test_list[0])
    np.random.seed(1337)

    length = len(train_val_test_list)

    train_val_list = np.random.choice(
        train_val_test_list, int(length * 0.95), replace=False)

    train_list = np.random.choice(
        train_val_list, int(length * 0.80), replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]
    test_list = [
        elem for elem in train_val_test_list if elem not in train_val_list]

    return {"train": train_list, "val": val_list, "test": test_list}
