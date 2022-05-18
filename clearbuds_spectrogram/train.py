import argparse
import os

import cv2
import random
import torch
import tqdm
import time

import numpy as np

from dataloader import SpatialAudioDataset
from UNet import unet
from CausalUNet import causal_unet

LEARNING_RATE = .001
WEIGHT_DECAY = 0
EPOCHS = 20

# np.random.seed(0)
# torch.manual_seed(0)
# random.seed(0)



def train(model, device, optimizer, train_loader, checkpoints_dir, clip_norm=0.5, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = []
    for batch_idx, (data, label) in enumerate(tqdm.tqdm(train_loader)):
        data, label = data.to(device), label.to(device)
        data = (data - data.mean()) / (data.std() + 1e-8)
        optimizer.zero_grad()

        if train:
            output = model(data)

            # Causal
            loss = model.weighted_binary_cross_entropy(output[:, :, :, -2:], label[:, :, :, -2:], weights=np.array([1, 1]))
            if torch.isnan(loss):
                print("skipping nan")
                import pdb
                pdb.set_trace()
                continue
            loss.backward()

            # Clip the gradient
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()

        else:
            with torch.no_grad():
                output = model(data)
                loss = model.weighted_binary_cross_entropy(output, label, weights=np.array([1, 1]))

        total_loss.append(loss.detach().cpu().numpy())

        # Check for Nan
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

        if np.random.uniform() < 0.1:

            to_write = output[0, 0].detach().cpu().numpy()
            cv2.imwrite("training.png", to_write * 255)

            input_frame = data[0, 0].detach().cpu().numpy()
            cv2.imwrite("input.png", input_frame * 255)

            target = label[0, 0].detach().cpu().numpy()
            cv2.imwrite("target.png", target * 255)
            
            print("Loss: {}".format(np.array(total_loss).mean()))

    print("Loss: {}".format(np.array(total_loss).mean()))
    torch.save(model, os.path.join(checkpoints_dir, "model_causal.pt"))



def main(args):
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    data_train = SpatialAudioDataset(args.data_train_path, n_mics=2, sr=args.sample_rate,
                                             target_fg_std=.03, target_bg_std=.03, perturb_prob=0.0,
                                             n_speakers=1, chunk_size=args.chunk_size)
    data_test = SpatialAudioDataset(args.data_test_path, n_mics=2, sr=args.sample_rate,
                                             target_fg_std=.03, target_bg_std=.03, perturb_prob=0.0,
                                             n_speakers=1, chunk_size=args.chunk_size)

    kwargs = {'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               **kwargs)

    model = unet().to(device)
    # model = causal_unet().to(device)
    if args.pretrain_path:
        state_dict = torch.load(args.pretrain_path).state_dict()
        model.load_pretrain(state_dict)


    for epoch in range(EPOCHS):
        print("Epoch: {}".format(epoch))
        optimizer = torch.optim.Adam(model.parameters(), lr=(LEARNING_RATE * (0.95 ** epoch)),
                                     weight_decay=WEIGHT_DECAY)

        train_loss = train(model, device, optimizer, train_loader, args.checkpoints_dir)
        test_loss = train(model, device, optimizer, test_loader, args.checkpoints_dir, train=False) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguments for network/main.py')
    parser.add_argument("data_train_path", type=str, help="Path to training samples")
    parser.add_argument("data_test_path", type=str, help="Path to testing samples")
    parser.add_argument("checkpoints_dir", type=str, help="Path to save model")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pretrain-path", type=str, help="Path to pretrained model")
    parser.add_argument("--sample-rate", type=int, help="Sample rate")
    parser.add_argument("--chunk-size", type=int, help="Number of samples to train with")
    main(parser.parse_args())
