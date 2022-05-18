import argparse
import json
import os
import random
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf

from conv_tasnet import ConvTasNet
from utils import get_piano_file, get_voice_file, inv_linear_quantize, si_sdr


DURATION = 48000


def run_once(args, idx, model):
    random.seed(idx)
    torch.manual_seed(idx)
    np.random.seed(idx)

    piano = get_piano_file(idx, DURATION, quantize_type=1)
    piano = inv_linear_quantize(piano, 255)
    voice = get_voice_file(idx, DURATION, quantize_type=1)
    voice = inv_linear_quantize(voice, 255)

    random_prefix = ''.join(random.sample(string.ascii_lowercase + string.digits, 12))
    # print(str(args["seed"]) + " " +random_prefix)
    writing_dir = os.path.join(args.writing_dir, random_prefix)
    if not os.path.exists(writing_dir):
        os.makedirs(writing_dir)
    print("writing dir: {}".format(writing_dir))

    with torch.no_grad():
        piano = torch.FloatTensor(piano).unsqueeze(0)
        voice = torch.FloatTensor(voice).unsqueeze(0)

        mixture = (piano + voice).unsqueeze(0).to(args.device)

        output_signal = model(mixture)

        # Un-normalize
        label_voice_signals = torch.stack((piano, voice), dim=1).to(args.device)


        gt0 = piano[0].numpy()
        est0 = output_signal[0, 0].cpu().numpy()
        est0 /= abs(est0).max()
        gt1 = voice[0].numpy()
        est1 = output_signal[0, 1].cpu().numpy()
        est1 /= abs(est1).max()

        sf.write(os.path.join(writing_dir, "gt0.wav"), gt0, 22050)
        sf.write(os.path.join(writing_dir, "gt1.wav"), gt1, 22050)
        sf.write(os.path.join(writing_dir, "out0.wav"), est0, 22050)
        sf.write(os.path.join(writing_dir, "out1.wav"), est1, 22050)

        sdr0 = si_sdr(est0, gt0)
        sdr1 = si_sdr(est1, gt1)

        with open(os.path.join(writing_dir, "info.json"), "w") as f:
            json.dump({"SDR0" : sdr0, "SDR1" : sdr1}, f, indent=4)
        print("SI SDR: {}, {}".format(sdr0, sdr1))
        return sdr0, sdr1


def main(args):
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    args.device = device

    # Load the model
    model = ConvTasNet.load_model(args.model_checkpoint).to(device)
    model.eval()

    assert(args.writing_dir is not None)

    sdr0 = []
    sdr1 = []
    for i in range(100):
        try:
            out0, out1 = run_once(args, i, model)
            sdr0.append(out0)
            sdr1.append(out1)
        except:
            print("Didn't work")

    print(np.median(np.array(sdr0)))
    print(np.median(np.array(sdr1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_checkpoint', type=str,
                        help="Path to the model file")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--writing_dir', type=str,
                        help="Where to write the output files")
    main(parser.parse_args())