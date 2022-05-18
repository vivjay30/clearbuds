#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os
import random
import time

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
from cached_model import CachedModel
from utils import remove_pad
import soundfile as sf

from irm import compute_irm

PADDING_AMOUNT = 24700
LOOKAHEAD = 700
R = 2
X = 7
N = 256


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
                    help='Duration of inference in seconds')


def evaluate(args):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    all_input_sdr = []
    all_output_sdr = []

    all_input_sar = []
    all_output_sar = []

    all_input_sir = []
    all_output_sir = []

    # Load model
    model = ConvTasNet.load_model(args.model_path, input_channels=args.n_mics)
    # model = ConvTasNet(N=256, L=40, B=256, H=512, P=3, X=8, R=4, C=1, input_channels=2)
    model.eval()
    if args.use_cuda:
        model.cuda()

    model = CachedModel(args.model_path, args.n_mics, N=N, use_cuda=args.use_cuda)

    import pdb
    pdb.set_trace()

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
            print(i)
            # Get batch data
            padded_mixture, mixture_lengths, padded_source = data
            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                padded_mixture = F.pad(padded_mixture, (PADDING_AMOUNT, 0))


            estimated_sources = []
            mixture = padded_mixture

            encoder_buffer = torch.zeros(1, N, 516).to(padded_mixture.device)
            conv_buffers = torch.zeros(R*X + 1, 1, N, 516).to(padded_mixture.device)

            while True:
                print("hey\n")
                t0 = time.time()
                if mixture.shape[2] <= 400:
                    break

                cur_mixture = mixture[:, :, :400]

                encoder_buffer = model.encoder_object(cur_mixture, encoder_buffer)
                conv_buffers = model.layernorm_object(encoder_buffer, conv_buffers)
                estimate_source = model.decoder_object(encoder_buffer, conv_buffers)
                estimated_sources.append(estimate_source.detach().cpu().numpy())
                mixture = mixture[:, :, 400:]

                print(time.time() - t0)

            # irm = torch.FloatTensor(compute_irm(padded_source[0].detach().cpu().numpy(), mixture[0].detach().cpu().numpy(), 1))[:, 0:1]
            # estimate_source = torch.tensor(np.concatenate(estimated_sources, axis=2))

            estimate_source = torch.tensor(np.concatenate(estimated_sources, axis=2))

            # import pdb
            # pdb.set_trace()
            # loss_0, max_snr_0, estimate_source_0, reorder_estimate_source_0 = \
            #     cal_loss(padded_source[:, :, 0, :-LOOKAHEAD], mixture[:, 0:1, -args.chunk_size:-LOOKAHEAD].detach().cpu(), mixture_lengths)

            # print("Input SDR {}".format(max_snr_0))

            # loss, max_snr, estimate_source, reorder_estimate_source = \
            #     cal_loss(padded_source[:, :, 0, :-LOOKAHEAD], estimate_source.detach().cpu(), mixture_lengths)

            # print("Output SDR {}".format(max_snr))

            # _, irm_snr, _, _= \
            #     cal_loss(padded_source[:, :, 0], irm, mixture_lengths)

            # print("IRM SDR {}".format(irm_snr))

            # # mixture_resampled = librosa.resample(mixture[0, 0,].detach().cpu().numpy(), args.sample_rate, 8000)
            # # gt_resampled = librosa.resample(padded_source[0, 0, 0].numpy(), args.sample_rate, 8000)
            # # output_resampled = librosa.resample(estimate_source[0, 0].numpy(), args.sample_rate, 8000)
            # # input_pesq = pesq.pesq(8000, gt_resampled, mixture_resampled, 'nb')
            # # print("Input PESQ {}".format(input_pesq))

            # # output_pesq = pesq.pesq(8000, gt_resampled, output_resampled, 'nb')
            # # print("Output PESQ {}".format(output_pesq))

            # all_input_sdr.append(max_snr_0)
            # all_output_sdr.append(max_snr)

            for batch_idx in range(len(estimate_source)):
                sf.write("evaluation_outputs/mixture.wav", padded_mixture.detach().cpu().permute(0, 2, 1).numpy()[0], args.sample_rate)
                
                for voice_idx in range(1):
                    output = estimate_source[0, 0].cpu().numpy()
                    # output = reorder_estimate_source[batch_idx, voice_idx].cpu().numpy()
                    # output /= np.abs(output).max()

                    sf.write("evaluation_outputs/output{}.wav".format(voice_idx), output, args.sample_rate)
                    sf.write("evaluation_outputs/gt{}.wav".format(voice_idx), padded_source.detach().cpu().numpy()[0, 0, 1], args.sample_rate)
                    
                    # sf.write("evaluation_outputs/irm.wav", irm[0, 0], args.sample_rate)
                    # input_sdr, input_sir, input_sar, input_perm = bss_eval_sources(padded_source[batch_idx, voice_idx, 0].cpu().numpy(),
                    #     padded_mixture[batch_idx, 0].cpu().numpy(), compute_permutation=False)

                    # output_sdr, output_sir, output_sar, output_perm = bss_eval_sources(padded_source[batch_idx, voice_idx, 0].cpu().numpy(), output, compute_permutation=False)

                    # all_input_sdr.append(input_sdr)
                    # all_input_sir.append(input_sir)
                    # all_input_sar.append(input_sar)

                    # all_output_sdr.append(output_sdr)
                    # all_output_sir.append(output_sir)
                    # all_output_sar.append(output_sar)

            import pdb
            pdb.set_trace()
    joint_data = np.stack((np.array(all_input_sdr), np.array(all_output_sdr)), axis=1)
    print("Median SI-SDR {}".format(np.median(joint_data[:, 1] - joint_data[:, 0])))
    # np.save("results/{}.npy".format(args.model_path.split("/")[-2]), joint_data)
    # np.save("results/{}_duplicate_channels.npy".format(args.model_path.split("/")[-2]), joint_data)
    # np.save("results/irm.npy".format(args.model_path.split("/")[-2]), joint_data)
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    evaluate(args)
