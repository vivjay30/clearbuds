import argparse
import os
import glob

import torch
import numpy as np

import librosa
import soundfile as sf

from clearbuds_spectrogram.UNet import unet


def save_spectrogram(spectrogram, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(filename, spectrogram)

def log_mel_spec_original(audio_files, sample_rate=None):
    audio = []
    for file in audio_files:
        y, sample_rate = librosa.core.load(file, sr=sample_rate)
        y = y / ((y.std() + 1e-4) / 0.05)
        audio.append(y)

    y = sum(audio)
    sf.write("mixed.wav", y, sample_rate)
    n_fft = 1024
    hop_length = 256
    n_mels = 128  # 128 is better for the direction part
    fmin = 20
    fmax = sample_rate / 2 

    original_spectrogram = librosa.stft(y, n_fft=n_fft)
    power_spectrogram = np.abs(original_spectrogram) ** 2
    S = librosa.feature.melspectrogram(S=power_spectrogram, sr=sample_rate, n_mels=n_mels,
                                       fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(S)

    return mel_spec_db, original_spectrogram


def generate_input(input_dir, sample_rate):
    """
    Generates the spectrograms for the mixed mic array inputs
    """
    gt_audio_l = os.path.join(input_dir, "mic00_voice00.wav")
    gt_audio_r = os.path.join(input_dir, "mic01_voice00.wav")

    # bg_voice_audio = os.path.join(input_dir, "mic00_voice01.wav")
    # bg_audio = os.path.join(input_dir, "mic00_bg.wav")
    mixed_specgrams = []
    original_specgrams = []
    # for i, mixed_audio_file in enumerate(mixed_audio_files):
    mixed_spec, original_spec = log_mel_spec_original([gt_audio_l, gt_audio_r], sample_rate=sample_rate)
    save_spectrogram(mixed_spec, os.path.join("mixed_spec.png"))
    # mixed_spec2, original_spec = log_mel_spec_original([os.path.join(input_dir, "output0.wav")], sample_rate=sample_rate)
    # save_spectrogram(mixed_spec2, os.path.join("mixed_spec2.png"))
    return mixed_spec, original_spec 


def infer(model, device, args):
    mixed_spec, original_spec = generate_input(args.input_dir, args.sample_rate)

    cropped_dim = (mixed_spec.shape[1] // 32) * 32
    mixed_spec = mixed_spec[:, :cropped_dim]
    original_spec = original_spec[:, :cropped_dim]

    mixed_spec = (mixed_spec - mixed_spec.mean()) / (mixed_spec.std() + 1e-8)

    output = model(torch.FloatTensor(mixed_spec).unsqueeze(0).unsqueeze(0).to(device))

    mask = output.cpu().detach().numpy()[0, 0]

    save_spectrogram(mask, os.path.join("mask.png"))
    # Undo the mel transform
    filter_bank = librosa.filters.mel(sr=args.sample_rate, n_fft=1024, n_mels=128, fmin=20, fmax=args.sample_rate / 2)
    spec_mask = np.matmul(filter_bank.transpose(), mask) > args.cutoff
    # Ratio mask from .003 - .006
    # spec_mask = (np.matmul(filter_bank.transpose(), mask) * 1000 / 3) - 1
    # spec_mask = np.clip(spec_mask, 0, 1)
    separated_spec = original_spec * spec_mask
    output = librosa.istft(separated_spec, hop_length = 256)

    sf.write(os.path.join(args.output_dir, "output.wav"), output, args.sample_rate)
    return output

def main(args):
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    model = unet().to(device)
    state_dict = torch.load(args.checkpoints_path).state_dict()
    model.load_pretrain(state_dict)

    model.eval()

    infer(model, device, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguments for network/main.py')
    parser.add_argument("input_dir", type=str, help="Path to testing samples")
    parser.add_argument("checkpoints_path", type=str, help="Path to save model")
    parser.add_argument("output_dir", type=str, help="Path to save model")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--sample-rate", type=int, help="Sample rate")
    parser.add_argument("--chunk-size", type=int, help="Number of samples to train with")
    parser.add_argument("--cutoff", type=float)

    main(parser.parse_args())