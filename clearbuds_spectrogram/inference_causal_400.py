import argparse
import os
import glob

import cv2
import torch
import numpy as np

import librosa
import soundfile as sf

from clearbuds_spectrogram.UNet import unet

TIME_DIM = 64
PROCESS_SIZE = 2

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

    min_length = min([x.shape[0] for x in audio])
    y = sum([x[:min_length] for x in audio])
    # sf.write("mixed.wav", y, sample_rate)

    return log_mel_spec(y, sample_rate)


def log_mel_spec(y, sample_rate):
    n_fft = 1024
    hop_length = 400
    window_length = 400
    n_mels = 128  # 128 is better for the direction part
    fmin = 20
    fmax = sample_rate / 2 

    original_spectrogram = librosa.stft(y, n_fft=n_fft,
                                        hop_length=hop_length,
                                        win_length=window_length)
    power_spectrogram = np.abs(original_spectrogram) ** 2
    S = librosa.feature.melspectrogram(S=power_spectrogram, sr=sample_rate, n_mels=n_mels,
                                       fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(S)

    return mel_spec_db, original_spectrogram

def apply_mask(mask, spectrogram):
    """
    """
    assert(mask.shape == spectrogram.shape)
    silence = np.ones_like(mask) * -100.0
    output = np.where(mask, spectrogram, silence)
    return output

def generate_input(input_dir, sample_rate, spectrogram_only=False):
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

    mixed_spec2, original_spec2 = None, None
    if not spectrogram_only:
        # We are doing a cascaded processing where the mask is applied to the time domain output
        mixed_spec2, original_spec2 = log_mel_spec_original([os.path.join(input_dir, "output0.wav")], sample_rate=sample_rate)
        save_spectrogram(mixed_spec2, os.path.join("mixed_spec2.png"))
    return mixed_spec, original_spec, mixed_spec2, original_spec2


def infer(model, device, args):
    mixed_spec, original_spec, mixed_spec2, original_spec2 = generate_input(args.input_dir, args.sample_rate, args.spectrogram_only)

    outputs = []

    curr_idx = 0
    mixed_spec_padded = np.pad(mixed_spec, ((0, 0), (TIME_DIM - PROCESS_SIZE, 0)))

    while (curr_idx + TIME_DIM) < mixed_spec_padded.shape[1]:
        curr_input = mixed_spec_padded[:, curr_idx:curr_idx+TIME_DIM]
        curr_input = (curr_input - curr_input.mean()) / (curr_input.std() + 1e-8)

        output = model(torch.FloatTensor(curr_input).unsqueeze(0).unsqueeze(0).to(device))

        mask = output.cpu().detach().numpy()[0, 0, :, -2:]
        outputs.append(mask)
        curr_idx += PROCESS_SIZE

    mask = np.concatenate(outputs, axis=-1)
    # save_spectrogram(mask > (args.cutoff * 1024 / 128), os.path.join("binary_mask_causal.png"))
    # binary_mask = (mask > (args.cutoff * 1024 / 16))
    # cv2.imwrite("binary_mask_causal.png", (binary_mask * 255).astype(np.uint8))
    cv2.imwrite("mask_causal.png", (mask * 255).astype(np.uint8))

    # Select noisy or conv-tasnet input
    apply_spectrogram_mel = mixed_spec if args.spectrogram_only else mixed_spec2
    apply_spectrogram_orig = original_spec if args.spectrogram_only else original_spec2

    total_length = min(mask.shape[-1], apply_spectrogram_orig.shape[-1])
    apply_spectrogram_orig = apply_spectrogram_orig[:, :total_length]
    mask = mask[:, :total_length]

    # Undo the mel transform
    filter_bank = librosa.filters.mel(sr=args.sample_rate, n_fft=1024, n_mels=128, fmin=20, fmax=args.sample_rate / 2)

    spec_mask = np.matmul(filter_bank.transpose(), mask) > args.cutoff
    # separated_spec = np.matmul(filter_bank.transpose(), apply_mask(binary_mask[:, :total_length], apply_spectrogram_mel[:, :total_length]))
    
    # Ratio mask from .003 - .006
    # spec_mask = (np.matmul(filter_bank.transpose(), mask) * 1000 / 3) - 1
    # spec_mask = np.clip(spec_mask, 0, 1)
    # separated_spec = apply_mask(spec_mask, apply_spectrogram_orig)
    separated_spec = spec_mask * apply_spectrogram_orig
    output = librosa.istft(separated_spec, hop_length=400, win_length=400)

    sf.write(os.path.join(args.output_dir, "output.wav"), output, args.sample_rate)
    output_spectrogram, _ = log_mel_spec(output, args.sample_rate)
    save_spectrogram(output_spectrogram, "output_spec.png")

    # save_spectrogram(apply_mask(binary_mask[:, :total_length], mixed_spec[:, :total_length]), "output_spec.png")


    gt_spectrogram, _ = log_mel_spec_original([os.path.join(args.input_dir, "gt.wav")])
    save_spectrogram(gt_spectrogram, "gt_spec.png")
    import pdb
    pdb.set_trace()
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
    parser.add_argument("--spectrogram-only", action="store_true")
    parser.add_argument("--cutoff", type=float)
    main(parser.parse_args())