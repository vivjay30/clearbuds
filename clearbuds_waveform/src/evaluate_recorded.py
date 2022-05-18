import argparse
import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F

# from data import AudioDataLoader, AudioDataset
from conv_tasnet import ConvTasNet
import soundfile as sf


LOOKAHEAD = 700
PADDING_AMOUNT = 25400 - LOOKAHEAD


LOOKAHEAD = 700
PADDING_AMOUNT = 25400 - LOOKAHEAD

def evaluate(args):
    # Load model
    model = ConvTasNet.load_model(args.model_path, input_channels=args.n_mics)
    model.eval()
    if args.use_cuda:
        model.cuda()

    full_path_l = args.file_path_left
    data_l = librosa.core.load(full_path_l, sr=args.sample_rate, mono=True)[0]
    full_path_r = full_path_l[:-5] + "R.wav"
    data_r = librosa.core.load(full_path_r, sr=args.sample_rate, mono=True)[0]
    min_length = min(data_l.shape[0], data_r.shape[0])
    data_l = data_l[:min_length]
    data_r = data_r[:min_length]

    data = np.stack([data_l, data_r], axis=0)

    mixture = torch.FloatTensor(data).unsqueeze(0)

    if args.use_cuda:
        mixture = mixture.cuda()

    
    mixture = F.pad(mixture, (PADDING_AMOUNT, 0))

    estimate_source = model(mixture)  # [B, C, T]
    output = estimate_source[0, 0].detach().cpu().numpy()
    sf.write("evaluation_outputs/output.wav", output, args.sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model file created by training')
    parser.add_argument('--use-cuda', type=int, default=1,
                        help='Whether use GPU')
    parser.add_argument('--sample-rate', default=8000, type=int,
                        help='Sample rate')
    parser.add_argument('--n-mics', default=2, type=int,
                        help='Number of mics')
    parser.add_argument("--file-path-left", type=str)
    args = parser.parse_args()
    evaluate(args)
