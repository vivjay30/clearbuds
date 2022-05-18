#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse

import torch

# from data import AudioDataLoader, AudioDataset
# from pianovoice_dataloader import PianoVoiceDataset
from our_data import SpatialAudioDatasetWaveform, MixedDataset, MixedDatasetDouble
from solver import Solver
from conv_tasnet import ConvTasNet


parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir_synth', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--train_dir_real', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
# parser.add_argument('--train_dir_real2', type=str, default=None,
#                     help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir_synth', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir_real', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=8, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
parser.add_argument('--n_mics', default=1, type=int,
                    help='Number of microphones')
# Network architecture
parser.add_argument('--N', default=256, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--L', default=40, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=256, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=512, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=4, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=2, type=int,
                    help='Number of speakers')
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')
# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='TasNet training',
                    help='Identifier for visdom run')
parser.add_argument('--chunk_size', default=None, type=int,
                    help="Training on small chunks of audio")


def main(args):
    # data_train = PianoVoiceDataset(sr=args.sample_rate, test=False)
    # data_test = PianoVoiceDataset(sr=args.sample_rate, test=True)
    data_train = MixedDataset(args.train_dir_synth, args.train_dir_real, n_mics=args.n_mics,
                                             sr=args.sample_rate,
                                             target_fg_std=None, target_bg_std=None,
                                             perturb_prob=0.6,
                                             n_speakers=2, chunk_size=args.chunk_size)
    # data_train = MixedDatasetDouble(args.train_dir_synth, args.train_dir_real, args.train_dir_real2, n_mics=args.n_mics,
    #                                          sr=args.sample_rate,
    #                                          target_fg_std=.03, target_bg_std=.03,
    #                                          perturb_prob=0.6,
    #                                          n_speakers=2, chunk_size=args.chunk_size)
    data_test = MixedDataset(args.valid_dir_synth, args.valid_dir_real, n_mics=args.n_mics,
                                            sr=args.sample_rate,
                                            target_fg_std=None, target_bg_std=None,
                                            perturb_prob=0.0,
                                            n_speakers=2, chunk_size=args.chunk_size)


    kwargs = {'num_workers': args.num_workers,
              'pin_memory': True} if args.use_cuda else {}

    # Set up data loaders
    tr_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=args.shuffle,
                                               **kwargs)
    cv_loader = torch.utils.data.DataLoader(data_test,
                                               batch_size=args.batch_size,
                                               shuffle=args.shuffle,
                                               **kwargs)

    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    model = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                       args.C, norm_type=args.norm_type, causal=args.causal,
                       mask_nonlinear=args.mask_nonlinear, input_channels=args.n_mics)

    # model = ConvTasNet.load_model("checkpoints/clearvoice_iphone_causal_mixed_l1spec_loss_large_real/final.pth.tar",
    #                               input_channels=args.n_mics)

    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

