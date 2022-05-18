import argparse
import os
import random
import math

from collections import namedtuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import pesq
import soundfile as sf

from our_data import SpatialAudioDatasetWaveform
from utils import si_sdr

from DTLN.DTLN_model import DTLN_model
from DTLN.run_evaluation import process_file


# Lookahead and padding in samples, carefully determined
LOOKAHEAD = 0
PADDING_AMOUNT = 0


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


def calculate_sisdri(args, padded_source, mixture, cascaded_output):
    # Input SDR
    if LOOKAHEAD != 0:
        input_sdr = si_sdr(mixture[0, 0, -args.chunk_size:-LOOKAHEAD].detach().cpu(), padded_source[0, 0, 0, :-LOOKAHEAD])

    else:
        input_sdr = si_sdr(mixture[0, 0, -args.chunk_size:].detach().cpu(), padded_source[0, 0, 0, :])

    # Output SDR
    output_len = cascaded_output.shape[0]
    if LOOKAHEAD != 0:
        output_sdr = si_sdr(torch.FloatTensor(cascaded_output), padded_source[0, 0, 0, :output_len])
    else:
        output_sdr = si_sdr(torch.FloatTensor(cascaded_output), padded_source[0, 0, 0, :])

    return input_sdr, output_sdr


def calculate_pesq(args, padded_source, mixture, cascaded_output):
    try:
        output_len = cascaded_output.shape[0]
        mixture_resampled = librosa.resample(mixture[0, 0].detach().cpu().numpy(), args.sample_rate, 8000)
        gt_resampled = librosa.resample(padded_source[0, 0, 0, :output_len].numpy(), args.sample_rate, 8000)
        output_resampled = librosa.resample(cascaded_output, args.sample_rate, 8000)
        
        input_pesq = pesq.pesq(8000, gt_resampled, mixture_resampled, 'nb')
        output_pesq = pesq.pesq(8000, gt_resampled, output_resampled, 'nb')

    # Errors when all zeros
    except:
        input_pesq = 0
        output_pesq = 0

    return input_pesq, output_pesq


def process_single_file(args, model, data):
    # Get batch data
    padded_mixture, mixture_lengths, padded_source = data
    
    padded_mixture = F.pad(padded_mixture, (PADDING_AMOUNT, 0))

    estimated_sources = []
    mixture = padded_mixture

    # This is slow because it writes it out for the spectrogram
    sf.write("evaluation_outputs/mic00_voice00.wav", mixture[0, 0, PADDING_AMOUNT:].detach().cpu().numpy(), args.sample_rate)
    # sf.write("evaluation_outputs/mic01_voice00.wav", mixture[0, 1, PADDING_AMOUNT:].detach().cpu().numpy(), args.sample_rate)
    # sf.write("evaluation_outputs/output0.wav", estimate_source[0, 0].detach().cpu().numpy(), args.sample_rate)
    # sf.write("evaluation_outputs/gt.wav", padded_source[0, 0, 0].detach().cpu().numpy(), args.sample_rate)

    process_file(model, "evaluation_outputs/mic00_voice00.wav", "evaluation_outputs/lstm.wav")

    lstm_output = librosa.core.load("evaluation_outputs/lstm.wav", sr=16000, mono=True)[0]

    input_sdr, output_sdr = calculate_sisdri(args, padded_source, mixture, lstm_output)
    print("Input SDR {}".format(input_sdr))
    print("Output SDR {}".format(output_sdr))

    input_pesq, output_pesq = calculate_pesq(args, padded_source, mixture, lstm_output)
    print("Input PESQ {}".format(input_pesq))
    print("Output PESQ {}".format(output_pesq))

    # # Write out the mixture so we can hear
    # sf.write("evaluation_outputs/mixture.wav",
    #          padded_mixture.detach().cpu().permute(0, 2, 1).numpy()[0, -args.chunk_size:-LOOKAHEAD],
    #          args.sample_rate)

    return input_sdr, output_sdr, input_pesq, output_pesq


def evaluate(args):
    all_input_sdr = []
    all_output_sdr = []

    all_input_pesq = []
    all_output_pesq = []

    # create class instance
    modelClass = DTLN_model();
    if args.model.find('_norm_') != -1:
        norm_stft = True
    else:
        norm_stft = False
    # build the model in default configuration
    modelClass.build_DTLN_model(norm_stft=norm_stft)
    # load weights of the .h5 file
    modelClass.model.load_weights(args.model)

    kwargs = {}

    data_test = SpatialAudioDatasetWaveform(args.data_dir,
                                            n_speakers=args.n_speakers,
                                            no_background=args.no_background,
                                            n_mics=args.n_mics,
                                            target_fg_std=0.03, target_bg_std=0.03,
                                            sr=args.sample_rate, perturb_prob=0.0,
                                            chunk_size=args.chunk_size)

    
    data_loader = torch.utils.data.DataLoader(data_test,
                                               batch_size=1,
                                               **kwargs)

    if len(data_loader) == 0:
        raise ValueError("No audiofiles found. Double check the data path argument.")
    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            print(i)
            input_sdr, output_sdr, input_pesq, output_pesq = process_single_file(args, modelClass.model, data)

            all_input_sdr.append(input_sdr)
            all_output_sdr.append(output_sdr)

            # Error case when both are zero
            if input_pesq != 0 or output_pesq !=0:
                all_input_pesq.append(input_pesq)
                all_output_pesq.append(output_pesq)                

            joint_data = np.stack((np.array(all_input_sdr), np.array(all_output_sdr)), axis=1)
            print("Median SI-SDR {}".format(np.median(joint_data[:, 1] - joint_data[:, 0])))

            print("Median PESQ {}".format(np.median(np.array(all_output_pesq))))

    np.save("results/lstm_sisdr.npy", joint_data)
    np.save("results/lstm_pesq.npy", np.stack((np.array(all_input_pesq), np.array(all_output_pesq)), axis=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='directory including mix.json, s1.json and s2.json')
    parser.add_argument('--model', '-m',
                        help='weights of the enhancement model in .h5 format')
    parser.add_argument('--sample-rate', default=16000, type=int,
                        help='Sample rate')
    parser.add_argument('--n-mics', default=2, type=int,
                        help='Number of mics')
    parser.add_argument('--n-speakers', default=2, type=int,
                        help='Number of mics')
    parser.add_argument('--chunk-size', default=None, type=int,
                        help='Duration of inference in samples')
    parser.add_argument('--no-background', action="store_true", default=False,
                        help='No background noise included')
    args = parser.parse_args()
    evaluate(args)
