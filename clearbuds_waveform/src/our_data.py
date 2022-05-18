import os
import json
import random

from typing import Tuple, Optional
from pathlib import Path

import torch
import numpy as np
import librosa
import soundfile as sf

from pysndfx import AudioEffectsChain

from rir import rir_perturbation

GLOBAL_SAMPLE_RATE=15625
SPEED_OF_SOUND = 343.0  # m/s
REVERB_VOICE = False


def check_valid_dir(dir):
    """Checks that there is at least a second voice"""
    if len(list(Path(dir).glob('*_voice01.wav'))) < 4:
        return False

    if len(list(Path(dir).glob('metadata.json'))) < 1:
        return False
    return True


class SpatialAudioDatasetWaveform(torch.utils.data.Dataset):
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
                 no_background=False,
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
        self.no_background = no_background

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
        keys = ["voice{:02}".format(i) for i in range(2)]
        keys.append("bg")
        assert(len(keys) == 3)  # 2 voices and a bg

        # Iterate over different sources
        all_sources = []
        target_voice_data = []
        for idx, key in enumerate(keys):
            # Skip the second speaker
            if self.n_speakers == 1 and idx == 1:
                continue
            # Skip the background
            if self.no_background and idx == 2:
                continue

            gt_audio_files = sorted(list(Path(curr_dir).rglob("*" + key + ".wav")))

            # Random levels
            if idx == 0:
                random_volume = np.random.uniform(2.0, 3.0)
            elif idx == 1:
                random_volume = np.random.uniform(1.0, 3.0)
            elif idx == 2:
                random_volume = np.random.uniform(1.0, 3.0)

            else:
                raise(ValueError("Idx {} Key {} Should be less than 2".format(idx, key)))

            # Just for a duplicate channel experiment
            # gt_audio_files[1] = gt_audio_files[0]
            assert(len(gt_audio_files) > 0)
            gt_waveforms = []
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

                gt_waveforms.append(torch.from_numpy(gt_waveform))
            #shifted_gt, _ = self.shift_input(torch.tensor(np.stack(gt_waveforms)).float(), np.array(locs_voice))

            if np.random.uniform() <= self.perturb_prob:
                perturbed_source = torch.tensor(random_perturb(np.stack(gt_waveforms)))
            else:
                perturbed_source = torch.tensor(np.stack(gt_waveforms))

            # Need to save for ground truth
            if "bg" not in key:
                target_voice_data.append(perturbed_source)

            if idx == 0 and REVERB_VOICE:
                reverb = np.random.uniform(low=0, high=100)
                fx = AudioEffectsChain().reverb(reverb)
                original_len = perturbed_source[0].shape[0]
                reverb_l = fx(perturbed_source[0].numpy())
                reverb_r = fx(perturbed_source[1].numpy())
                sf.write("reverbl.wav", reverb_l, self.sr)
                sf.write("reverbr.wav", reverb_r, self.sr)
                all_sources.append(torch.tensor(np.stack([reverb_l[:original_len], reverb_r[:original_len]])))
            else:
                all_sources.append(perturbed_source)


        all_sources = torch.stack(all_sources, dim=0)
        mixed_data = torch.sum(all_sources, dim=0)  # n_mics x t

        target_voice_data = torch.stack(target_voice_data[:1])  # n_speakers x n_mics x t


        if self.chunk_size:
            duration_samples = self.chunk_size # int(self.chunk_size * self.sr)
            random_start = np.random.randint(0, mixed_data.shape[1] - duration_samples - 1)
            mixed_data = mixed_data[:, random_start:random_start+duration_samples]
            target_voice_data = target_voice_data[:, :, random_start:random_start+duration_samples]

        with open(os.path.join(curr_dir, "metadata.json")) as f:
            metadata = json.load(f)

        # bg_position = torch.FloatTensor(metadata["voice01"]["position"])

        ilens = torch.tensor(mixed_data.shape[1])
        return (mixed_data, ilens, target_voice_data)


class RealDataset(torch.utils.data.Dataset):
    """
    Dataset of synthetic composites of real data
    """

    def __init__(self, input_dir, duration=3.0,
                 sr=GLOBAL_SAMPLE_RATE,
                 num_elements=1000,
                 perturb_prob=0.0,
                 short_data=False,
                 target_fg_std=0.03,
                 target_bg_std=0.04,
                 max_num_voices=3):
        super().__init__()
        self.duration = duration
        self.sr = sr
        self.fgs = []
        self.bgs = []
        self.num_elements = num_elements
        self.perturb_prob = perturb_prob
        self.max_num_voices = max_num_voices

        duration = 60.0 if short_data else None 

        # Read fg files
        all_fg_files = os.listdir(os.path.join(input_dir, "fg"))
        for fg_file in all_fg_files:
            full_path = os.path.join(input_dir, "fg", fg_file)
            data = librosa.core.load(full_path, sr=self.sr, mono=False, duration=duration)[0]
            shift = int(fg_file.split("shift")[-1][:-4])
            print(shift)
            data = data / (data.std() / target_fg_std)
            self.fgs.append((data, shift))

        # Read bg files
        all_bg_files = os.listdir(os.path.join(input_dir, "bg"))
        for bg_file in all_bg_files:
            full_path = os.path.join(input_dir, "bg", bg_file)
            data = librosa.core.load(full_path, sr=self.sr, mono=False, duration=duration)[0]
            data = data / (data.std() / target_bg_std)
            self.bgs.append(data)


    def __len__(self) -> int:
        return self.num_elements

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Choose the number of voices in the sample
        num_voices = random.randint(1, min(self.max_num_voices, len(self.fgs)))
        voices = random.sample(self.fgs, num_voices)

        # Randomly sample n voices
        all_fg = []
        all_shifts = []
        for curr_fg, curr_shift in voices:
            start_fg_sample = np.random.randint(0, curr_fg.shape[1] - (self.sr * self.duration + 1))
            random_fg_volume = np.random.uniform(0.5, 4.0)
            fg_data = curr_fg[:, start_fg_sample:int(start_fg_sample+(self.sr * self.duration))] * random_fg_volume
            all_fg.append(fg_data)
            all_shifts.append(curr_shift)

        # print("-----------------------------")
        # print("Shifts: {}".format(all_shifts))
        fg_data = sum(all_fg)
        
        # Sample 1 bg
        curr_bg = random.choice(self.bgs) 
        start_bg_sample = np.random.randint(0, curr_bg.shape[1] - (self.sr * self.duration + 1))
        random_bg_volume = np.random.uniform(0.1, 1.5)
        bg_data = curr_bg[:, start_bg_sample:int(start_bg_sample+(self.sr * self.duration))] * random_bg_volume
        random_bg_shift = np.random.randint(-12, 12)
        # print("BG Shift: {}".format(random_bg_shift))
        bg_data = self.shift_input(torch.tensor(bg_data), random_bg_shift).numpy()


        # Data augmentation
        if np.random.uniform() < self.perturb_prob:
            random_perturb = RandomAudioPerturbation()
            fg_data = random_perturb(fg_data)
            bg_data = random_perturb(bg_data)

        mixed_data = fg_data + bg_data
        shift = round(all_shifts[0] * self.sr / 44100)

        mixed_data = torch.tensor(mixed_data)
        mixed_data = self.shift_input(mixed_data, shift)

        gt = self.shift_input(torch.tensor(all_fg[0]), shift)

        return (mixed_data, gt.view(1, 2, -1))

    def shift_input(self, data, shift):
        """
        Shifts the input according to the voice position. This tried to
        line up the voice samples in the time domain
        # """
        data[0] = torch.roll(data[0], int(shift))

        return data

if __name__ == '__main__':
    data_train = SpatialAudioDatasetWaveform('/projects/grail/audiovisual/datasets/DinTaiFung/mics_8_radius_3_voice_1_bg_1/train')
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=4)

    x = None
    for x in train_loader:
        print(x)
        break

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, input_dir_synth, input_dir_real,
                 n_mics=1,
                 n_speakers=1,
                 n_backgrounds=1,
                 sr=GLOBAL_SAMPLE_RATE,
                 target_fg_std=0.03,
                 target_bg_std=0.04,
                 perturb_prob=0.0,
                 chunk_size=None):
        
        self.synth_dataset = SpatialAudioDatasetWaveform(input_dir_synth,
            n_mics=n_mics, sr=sr, target_fg_std=target_fg_std, target_bg_std=target_bg_std,
            perturb_prob=perturb_prob, n_speakers=n_speakers, chunk_size=chunk_size)

        self.real_dataset = SpatialAudioDatasetWaveform(input_dir_real,
            n_mics=n_mics, sr=sr, target_fg_std=target_fg_std, target_bg_std=target_bg_std,
            perturb_prob=perturb_prob, n_speakers=n_speakers, chunk_size=chunk_size)

    def __len__(self):
        return len(self.synth_dataset) + len(self.real_dataset)

    def __getitem__(self, idx: int):
        if idx < len(self.real_dataset):
            return self.real_dataset[idx]
        else:
            return self.synth_dataset[idx - len(self.real_dataset)]

class MixedDatasetDouble(torch.utils.data.Dataset):

    def __init__(self, input_dir_synth, input_dir_real, input_dir_real2,
                 n_mics=1,
                 n_speakers=1,
                 n_backgrounds=1,
                 sr=GLOBAL_SAMPLE_RATE,
                 target_fg_std=0.03,
                 target_bg_std=0.04,
                 perturb_prob=0.0,
                 chunk_size=None):
        
        self.synth_dataset = SpatialAudioDatasetWaveform(input_dir_synth,
            n_mics=n_mics, sr=sr, target_fg_std=target_fg_std, target_bg_std=target_bg_std,
            perturb_prob=perturb_prob, n_speakers=n_speakers, chunk_size=chunk_size)

        self.real_dataset = SpatialAudioDatasetWaveform(input_dir_real,
            n_mics=n_mics, sr=sr, target_fg_std=target_fg_std, target_bg_std=target_bg_std,
            perturb_prob=perturb_prob, n_speakers=n_speakers, chunk_size=chunk_size)

        self.real_dataset2 = SpatialAudioDatasetWaveform(input_dir_real2,
            n_mics=n_mics, sr=sr, target_fg_std=target_fg_std, target_bg_std=target_bg_std,
            perturb_prob=perturb_prob, n_speakers=n_speakers, chunk_size=chunk_size)

    def __len__(self):
        return len(self.synth_dataset) + len(self.real_dataset) + len(self.real_dataset2)

    def __getitem__(self, idx: int):
        if idx < len(self.real_dataset):
            return self.real_dataset[idx]
        elif idx < (len(self.real_dataset) + len(self.synth_dataset)):
            return self.synth_dataset[idx - len(self.real_dataset)]
        else:
            return self.real_dataset2[idx - (len(self.real_dataset) + len(self.synth_dataset))]

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

