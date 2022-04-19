from glob import glob
import os
import random

import h5py
import numpy as np
from sortedcontainers import SortedList
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data.utils import get_overlay_ratio, load, mask_audio, overlay_bgm


class SeparationDataset(Dataset):
    def __init__(self, dataset, partition, sr, channels, shapes, random_hops, hdf_dir, bgm_dataset, in_memory=False, augument_ratio=1):
        '''
        Initialises a source separation dataset
        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(SeparationDataset, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")

        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.in_memory = in_memory
        self.augument_ratio = augument_ratio
        self.bgm_path_list = bgm_dataset[partition]

        # PREPARE HDF FILE

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_dir):
            # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_dir, "w") as f:
                f.attrs["sr"] = sr
                f.attrs["channels"] = channels

                print("Adding audio files to dataset (preprocessing)...")
                for idx, example in enumerate(tqdm(dataset[partition])):
                    # Load mix
                    audio, _ = load(
                        example, sr=self.sr, mono=(self.channels == 1))

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset(
                        "inputs", shape=audio.shape, dtype=audio.dtype, data=audio)
                    grp.attrs["length"] = audio.shape[1]

                print("Adding bgm files to dataset (preprocessing)...")
                for idx, bgm_path in enumerate(tqdm(self.bgm_path_list)):
                    # Load bgm
                    audio, _ = load(bgm_path, sr=self.sr,
                                    mono=(self.channels == 1))

                    # Add to HDF5 file
                    grp = f.create_group('bgm-'+str(idx))
                    grp.create_dataset(
                        "bgm", shape=audio.shape, dtype=audio.dtype, data=audio)

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_dir, "r") as f:
            if f.attrs["sr"] != sr or \
                    f.attrs["channels"] != channels:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate and channel are not as expected. Did you load an out-dated HDF file?")

        # HDF FILE READY

        self.length = len(dataset[partition]) * self.augument_ratio

    def __getitem__(self, index):
        # Open HDF5
        if self.hdf_dataset is None:
            # Load HDF5 fully into memory if desired
            driver = "core" if self.in_memory else None
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r', driver=driver)

        # Find out which slice of targets we want to read
        audio_idx = index // self.augument_ratio

        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]

        # TODO: スピード調整, audio_length を変更する

        # 切り抜く
        if audio_length > self.shapes["output_frames"]:
            start_target_pos = np.random.randint(
                0, max(audio_length - self.shapes["output_frames"] + 1, 1))
        else:
            start_target_pos = 0

        # READ INPUTS
        # Check front padding
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - \
            self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # read and padding
        audio = self.hdf_dataset[str(
            audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)  # type: ignore
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)],
                           mode="constant", constant_values=0.0)

        audio = torch.from_numpy(audio)

        # mask 前にratioだけ決めとく

        bgm_index = random.randint(0, len(self.bgm_path_list) - 1)
        bgm = torch.from_numpy(
            np.copy(self.hdf_dataset['bgm-'+str(bgm_index)]["bgm"].astype(np.float32)))  # type: ignore
        bgm_ratio = get_overlay_ratio(audio, bgm)

        # TODO: mask 処理でinputをつくる
        masked_audio = mask_audio(
            audio, 'silent', (audio_length // 2) + start_pos, audio_length, self.sr)  # type: ignore

        # BGM
        # TODO: BGMありなしの割合
        if random.randint(0, 9) != 9:
            audio = overlay_bgm(
                audio, bgm=bgm, ratio=bgm_ratio)
            masked_audio = overlay_bgm(
                masked_audio, bgm=bgm, ratio=bgm_ratio)

        # crop output
        audio = audio[:, self.shapes["output_start_frame"]:self.shapes["output_end_frame"]]

        return masked_audio, audio

    def __len__(self):
        return self.length
