#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os
import random

import librosa
# from mir_eval.separation import bss_eval_sources
import numpy as np
import torch
import torch.nn.functional as F

import pesq

# from data import AudioDataLoader, AudioDataset
from our_data import SpatialAudioDatasetWaveform
from pit_criterion import cal_loss
from conv_tasnet import ConvTasNet
from utils import remove_pad
import soundfile as sf
import pyroomacoustics as pra

from irm import compute_irm
from ibm import compute_ibm

LOOKAHEAD = 700
PADDING_AMOUNT = 25400 - LOOKAHEAD
absorption = 0.1
Fs = 15625

# Define the FFT length
N = 1024

Lg_t = 0.100  # Filter size in seconds
Lg = np.ceil(Lg_t * Fs)  # Filter size in samples
max_order_sim = 2
sigma2_n = 5e-7
delay = 0.050  # Beamformer delay in seconds
delay1 = 0.0
delay2 = 0.0

parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=str, required=True,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')
parser.add_argument('--n_mics', default=1, type=int,
                    help='Number of mics')
parser.add_argument('--n_speakers', default=2, type=int,
                    help='Number of mics')
parser.add_argument('--chunk_size', default=None, type=int,
                    help='Duration of inference in samples')

R = pra.circular_2D_array(center=[0., 0.], M=2, phi0=0, radius=.0725)
mics = pra.Beamformer(R, Fs, N=N, Lg=Lg)

# Create the room
room_dim = [4, 6]
room1 = pra.ShoeBox(
    room_dim,
    absorption=absorption,
    fs=Fs,
    max_order=max_order_sim,
    sigma2_awgn=sigma2_n,
)




def evaluate(args):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    all_input_sdr = []
    all_output_sdr = []


    kwargs = {'num_workers': 0,
              'pin_memory': True} if args.use_cuda else {}

    data_test = SpatialAudioDatasetWaveform(args.data_dir,
                                            n_speakers=args.n_speakers,
                                            n_mics=args.n_mics,
                                            target_fg_std=0.03, target_bg_std=0.03,
                                            sr=args.sample_rate, perturb_prob=0.0,
                                            chunk_size=args.chunk_size)

    
    data_loader = torch.utils.data.DataLoader(data_test,
                                               batch_size=args.batch_size,
                                               **kwargs)

    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            padded_mixture, mixture_lengths, padded_source, bg_position = data

            clean_voice = padded_source[0, 0, 0].numpy()
            # Add sources to room
            good_source = np.array([1, 4.5])  # good source
            normal_interferer = np.array([2.8, 4.3])  # interferer
            room1.add_source(good_source, signal=clean_voice, delay=delay1)
            room1.add_source(normal_interferer, signal=clean_voice, delay=delay2)

            """
            MVDR direct path only simulation
            """

            # compute beamforming filters
            mics = pra.Beamformer(R, Fs, N=N, Lg=Lg)
            room1.add_microphone_array(mics)
            room1.compute_rir()
            room1.simulate()
            mics.rake_mvdr_filters(
                room1.sources[0][0:1],
                room1.sources[1][0:1],
                sigma2_n * np.eye(mics.Lg * mics.M),
                delay=delay,
            )

            # process the signal
            output = mics.process()

            # save to output file
            input_mic = pra.normalize(pra.highpass(mics.signals[mics.M // 2], Fs))
            sf.write("evaluation_outputs/input.wav", input_mic, Fs)

            out_DirectMVDR = pra.normalize(pra.highpass(output, Fs))
            sf.write("evaluation_outputs/MVDR.wav", out_DirectMVDR, Fs)

            import pdb
            pdb.set_trace()

            if LOOKAHEAD != 0:
                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source[:, :, 0, :-LOOKAHEAD], estimate_source.detach().cpu(), mixture_lengths)
            else:
                loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source[:, :, 0, :], estimate_source.detach().cpu(), mixture_lengths)

            print("Output SDR {}".format(max_snr))

            # angle = np.abs(np.arctan(metadata[0, 1] / metadata[0, 0]) * 180.0 / np.pi)
            # print("Angle {}".format(angle))

            irm = torch.FloatTensor(compute_irm(padded_source[0, :, :, :-LOOKAHEAD].detach().cpu().numpy(), mixture[0, :, -args.chunk_size:-LOOKAHEAD].detach().cpu().numpy(), 1))[:, 0:1]
            _, irm_snr, _, _= \
                cal_loss(padded_source[:, :, 0, :-LOOKAHEAD], irm, mixture_lengths)

            # print("IRM SDR {}".format(irm_snr))

            # # mixture_resampled = librosa.resample(mixture[0, 0,].detach().cpu().numpy(), args.sample_rate, 8000)
            # # gt_resampled = librosa.resample(padded_source[0, 0, 0].numpy(), args.sample_rate, 8000)
            # # output_resampled = librosa.resample(estimate_source[0, 0].numpy(), args.sample_rate, 8000)
            # # input_pesq = pesq.pesq(8000, gt_resampled, mixture_resampled, 'nb')
            # # print("Input PESQ {}".format(input_pesq))

            # # output_pesq = pesq.pesq(8000, gt_resampled, output_resampled, 'nb')
            # # print("Output PESQ {}".format(output_pesq))

            # if np.isnan(irm_snr):
            #     continue
            all_input_sdr.append(max_snr_0)
            all_output_sdr.append(max_snr)
            all_positions.append(bg_position[0].numpy())
            # all_angles.append(angle.detach().cpu().numpy())
            # all_output_sdr.append(irm_snr)

            for batch_idx in range(len(estimate_source)):
                sf.write("evaluation_outputs/mixture.wav", padded_mixture.detach().cpu().permute(0, 2, 1).numpy()[0, -args.chunk_size:-LOOKAHEAD], args.sample_rate)
                
                for voice_idx in range(1):
                    output = estimate_source[0, 0].cpu().numpy()
                    output = reorder_estimate_source[batch_idx, voice_idx].cpu().numpy()
                    # output /= np.abs(output).max()

                    # final_output = output * 0.95 + mixture[:, 0:1, -args.chunk_size:-LOOKAHEAD].detach().cpu().numpy()[0, 0] * 0.05
                    sf.write("evaluation_outputs/output{}.wav".format(voice_idx), output, args.sample_rate)
                    sf.write("evaluation_outputs/gt{}.wav".format(voice_idx), padded_source.detach().cpu().numpy()[0, 0, 1], args.sample_rate)
                    
                    sf.write("evaluation_outputs/irm.wav", irm[0, 0], args.sample_rate)
                    # input_sdr, input_sir, input_sar, input_perm = bss_eval_sources(padded_source[batch_idx, voice_idx, 0].cpu().numpy(),
                    #     padded_mixture[batch_idx, 0].cpu().numpy(), compute_permutation=False)

                    # output_sdr, output_sir, output_sar, output_perm = bss_eval_sources(padded_source[batch_idx, voice_idx, 0].cpu().numpy(), output, compute_permutation=False)

                    # all_input_sdr.append(input_sdr)
                    # all_input_sir.append(input_sir)
                    # all_input_sar.append(input_sar)

                    # all_output_sdr.append(output_sdr)
                    # all_output_sir.append(output_sir)
                    # all_output_sar.append(output_sar)

            joint_data = np.stack((np.array(all_input_sdr), np.array(all_output_sdr)), axis=1)
            print("Median SI-SDR {}".format(np.median(joint_data[:, 1] - joint_data[:, 0])))

    # np.save("results/{}.npy".format(args.model_path.split("/")[-2]), joint_data)
    # np.save("results/{}_duplicate_channels.npy".format(args.model_path.split("/")[-2]), joint_data)
    # np.save("results/irm.npy".format(args.model_path.split("/")[-2]), joint_data)
    import pdb
    pdb.set_trace()
    si_sdr = joint_data[:, 1] - joint_data[:, 0]
    # np.save("results/angle_graph.npy", np.stack((np.array(all_angles), si_sdr), axis=1))
    np.save("results/position_graph.npy", np.concatenate((np.array(all_positions), joint_data), axis=1))
    print(joint_data)
    print(all_angles)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    evaluate(args)
