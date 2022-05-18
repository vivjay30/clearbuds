import os
import json
import random

from typing import Tuple, Optional
from pathlib import Path

import torch
import numpy as np
import librosa

from pysndfx import AudioEffectsChain



GLOBAL_SAMPLE_RATE: int = 8000
SPEED_OF_SOUND = 343.0  # m/s


def check_valid_dir(dir):
    """Checks that there is at least a second voice"""
    if len(list(Path(dir).glob('*_voice01.wav'))) < 4:
        return False

    if len(list(Path(dir).glob('metadata.json'))) < 1:
        return False
    return True


def log_mel_spec_tfm(audio, sample_rate):
    """
    Generates a mel spectrogram with dB magnitude
    """
    n_fft = 400
    hop_length = 400
    window_length = 400
    n_mels = 128  # 128 is better for the direction part
    fmin = 20
    fmax = sample_rate / 2 
    
    mel_spec_power = librosa.feature.melspectrogram(audio, sr=sample_rate, n_fft=n_fft, 
                                                    hop_length=hop_length, 
                                                    win_length=window_length,
                                                    n_mels=n_mels, power=2.0, 
                                                    fmin=fmin, fmax=fmax, center=False)
    mel_spec_db = librosa.power_to_db(mel_spec_power)
    return mel_spec_db

class SpatialAudioDataset(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).
    """

    def __init__(self, input_path,
                 n_mics=1,
                 n_speakers=1,
                 n_backgrounds=1,
                 sr=GLOBAL_SAMPLE_RATE,
                 target_fg_std=0.03,
                 target_bg_std=0.04,
                 perturb_prob=0.0,
                 chunk_size=None):
        super().__init__()
        self.dirs = sorted(list(Path(input_path).glob('[0-9]*')))
        self.n_speakers = n_speakers
        self.n_backgrounds = n_backgrounds
        self.sr = sr
        self.target_fg_std = target_fg_std
        self.target_bg_std = target_bg_std
        self.perturb_prob = perturb_prob
        self.n_mics = n_mics
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        curr_dir = self.dirs[idx]

        # Get metadata
        with open(Path(curr_dir) / 'metadata.json') as json_file:
            json_data = json.load(json_file)

        num_voices = self.n_speakers

        mic_files = sorted(list(Path(curr_dir).rglob('*mixed.wav')))
        random_perturb = RandomAudioPerturbation()

        # GT voice signals
        keys = ["voice{:02}".format(i) for i in range(num_voices)]
        keys.append("bg")
        # assert(len(keys) == 3)  # 2 voices and a bg

        # Iterate over different sources
        all_sources = []
        target_voice_data = []
        for idx, key in enumerate(keys):
            # if idx == 1:
            #     continue
            # if idx > 0:
                # Randomly exclude background or second voice
            #     if np.random.uniform() < 0.3:
            #         continue

            gt_audio_files = sorted(list(Path(curr_dir).rglob("*" + key + ".wav")))
            # random.shuffle(gt_audio_files)

            if idx == 0:
                random_volume = np.random.uniform(2.0, 4.0)
            elif idx == 1:
                random_volume = np.random.uniform(2.0, 4.0)
            elif idx == 2:
                random_volume = np.random.uniform(2.0, 4.0)
            # REMOVE!
            # random_volume = 1.0

            # Just for a duplicate channel experiment
            # gt_audio_files[1] = gt_audio_files[0]
            assert(len(gt_audio_files) > 0)
            gt_waveforms = []
            bg_waveforms = []
            # Iterate over different mics
            for mic_idx, gt_audio_file in enumerate(gt_audio_files):
                gt_waveform, _ = librosa.core.load(gt_audio_file, self.sr, mono=True)

                # Normalize volume
                if "bg" not in key:
                    if self.target_fg_std is not None:
                        gt_waveform = gt_waveform / ((gt_waveform.std() + 1e-4) / self.target_fg_std)
                        gt_waveform *= random_volume
                else:
                    if self.target_bg_std is not None:
                        gt_waveform = gt_waveform / ((gt_waveform.std() + 1e-4) / self.target_bg_std)
                        gt_waveform *= random_volume
                        bg_waveforms.append(gt_waveform)

                gt_waveforms.append(torch.from_numpy(gt_waveform))
            #shifted_gt, _ = self.shift_input(torch.tensor(np.stack(gt_waveforms)).float(), np.array(locs_voice))

            if np.random.uniform() <= self.perturb_prob:
                perturbed_source = torch.tensor(random_perturb(np.stack(gt_waveforms)))
            else:
                perturbed_source = torch.tensor(np.stack(gt_waveforms))

            all_sources.append(perturbed_source)

            # Need to save for ground truth
            if "bg" not in key:
                target_voice_data.append(perturbed_source)
        
        all_sources = torch.stack(all_sources, dim=0)
        mixed_data = torch.sum(all_sources, dim=0)  # n_mics x t

        target_voice_data = torch.stack(target_voice_data[:1])  # n_speakers x n_mics x t


        if self.chunk_size:
            duration_samples = self.chunk_size # int(self.chunk_size * self.sr)
            random_start = np.random.randint(0, mixed_data.shape[1] - duration_samples - 1)
            mixed_data = mixed_data[:, random_start:random_start+duration_samples]
            target_voice_data = target_voice_data[:, :, random_start:random_start+duration_samples]
            bg_waveform = bg_waveforms[0][random_start:random_start+duration_samples]

        spec_gt = log_mel_spec_tfm(target_voice_data.numpy()[0, 0, :], self.sr)
        spec_mixture = log_mel_spec_tfm(mixed_data.numpy()[0, :], self.sr)
        spec_bg = log_mel_spec_tfm(bg_waveform, self.sr)

        return torch.FloatTensor(spec_mixture).unsqueeze(0), torch.FloatTensor(spec_gt > spec_bg).unsqueeze(0)
        # with open(os.path.join(curr_dir, "metadata.json")) as f:
        #     metadata = json.load(f)

        # bg_position = torch.FloatTensor(metadata["voice01"]["position"])

        # ilens = torch.tensor(mixed_data.shape[1])
        # return (mixed_data, ilens, target_voice_data, bg_position)


class RandomAudioPerturbation(object):
    """Randomly perturb audio samples"""

    def __call__(self, data):
        highshelf_gain = np.random.normal(0, 5)
        lowshelf_gain = np.random.normal(0, 5)
        noise_amount = np.random.uniform(0, 0.003)
        shift = random.randint(-1, 1)

        fx = (
            AudioEffectsChain()
            .highshelf(gain=highshelf_gain)
            .lowshelf(gain=lowshelf_gain)
        )

        for i in range(data.shape[0]):
            data[i] = fx(data[i])
            data[i] += np.random.uniform(-noise_amount, noise_amount, size=data[i].shape)
            np.roll(data[i], shift, axis=0)
        return data

