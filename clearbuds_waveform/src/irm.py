import argparse
import json
import multiprocessing.dummy as mp
import os
import math

from pathlib import Path

import librosa
import numpy as np

from scipy.signal import stft, istft

def si_sdr(estimated_signal, reference_signals, scaling=True):
    """
    This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
    or https://github.com/sigsep/bsseval/issues/3 for the motivation and
    explanation
    Input:
        estimated_signal and reference signals are (N,) numpy arrays
    Returns: SI-SDR as scalar
    """
    Rss = np.dot(reference_signals, reference_signals)
    this_s = reference_signals

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    SDR = 10 * math.log10(Sss/Snn)

    return SDR


def compute_sdr(gt, output, single_channel=False):
    assert(gt.shape == output.shape)
    per_channel_sdr = []

    channels = [0] if single_channel else range(gt.shape[0])
    for channel_idx in channels:
        # sdr, _, _, _ = bss_eval_sources(gt[channel_idx], output[channel_idx])
        sdr = si_sdr(output[channel_idx], gt[channel_idx])
        per_channel_sdr.append(sdr)

    return np.array(per_channel_sdr).mean()


def compute_irm(gt, mix, alpha):
    """
    Computes the Ideal Ratio Mask SI-SDR
    gt: (n_voices, n_channels, t)
    mix: (n_channels, t)
    """
    n_voices = gt.shape[0]
    nfft = 2048
    hop = 1024
    eps = np.finfo(np.float).eps

    N = mix.shape[-1] # number of samples
    X = stft(mix, nperseg=nfft)[2]
    (I, F, T) = X.shape # (6, nfft//2 +1, n_frame)

    # Compute sources spectrograms
    P = []
    for gt_idx in range(n_voices):
        P.append(np.abs(stft(gt[gt_idx], nperseg=nfft)[2]) ** alpha)
        
    # compute model as the sum of spectrograms
    model = eps
    # for gt_idx in range(n_voices):
    #     model += P[gt_idx]
    model = np.abs(stft(mix, nperseg=nfft)[2]) ** alpha

    # perform separation
    estimates = []
    for gt_idx in range(n_voices):
        # Create a ratio Mask
        mask = np.divide(np.abs(P[gt_idx]), model)
        
        # apply mask
        Yj = np.multiply(X, mask)

        target_estimate = istft(Yj)[1][:,:N]

        estimates.append(target_estimate)

    estimates = np.array(estimates) # (nvoice, 6, 6*sr)

    # eval
    eval_mix = np.repeat(mix[np.newaxis, :, :], n_voices, axis=0) # (nvoice, 6, 6*sr)
    eval_gt = gt # (nvoice, 6, 6*sr)
    eval_est = estimates

    SDR_in = []
    SDR_out = []
    return eval_est
    # for i in range(n_voices):
    #     SDR_in.append(compute_sdr(eval_gt[i], eval_mix[i], single_channel=True)) # scalar
    #     SDR_out.append(compute_sdr(eval_gt[i], eval_est[i], single_channel=True)) # scalar

    # output = np.array([SDR_in, SDR_out]) # (2, nvoice)

    # return output
