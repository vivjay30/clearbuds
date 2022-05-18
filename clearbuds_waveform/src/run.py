import os

import numpy as np
import torch
import librosa

import soundfile as sf

from conv_tasnet import ConvTasNet

MODEL_PATH = "/projects/grail/vjayaram/Conv-TasNet/src/checkpoints/4_mics_8000_nobg/final.pth.tar"
DATA = "/projects/grail/vjayaram/DinTaiFung/respeaker/2_person_moving.wav"
SR = 8000
DURATION = 3.0




def process_data(model, data):
    data = torch.tensor(data).cuda().unsqueeze(0)

    estimate_source = model(data)
    data1 = estimate_source[0,0].detach().cpu().numpy()
    data2 = estimate_source[0,1].detach().cpu().numpy()

    data1 /= (2*data1.max())
    data2 /= (2*data2.max())

    return data1, data2

    sf.write("data1.wav", data1, SR)
    sf.write("data2.wav", data2, SR)


model = ConvTasNet.load_model(MODEL_PATH, input_channels=4).cuda()
model.eval()

data, _ = librosa.core.load(DATA, mono=False, sr=SR)

temporal_chunk_size = int(SR * DURATION)
num_chunks = (data.shape[1] // temporal_chunk_size) + 1

for chunk_idx in range(num_chunks):
    curr_writing_dir = os.path.join("outputs",
                                    "{:03d}".format(chunk_idx))
    if not os.path.exists(curr_writing_dir):
        os.makedirs(curr_writing_dir)

    curr_data = data[:, (chunk_idx * temporal_chunk_size):(chunk_idx + 1) * temporal_chunk_size]

    output1, output2 = process_data(model, curr_data)

    sf.write(os.path.join(curr_writing_dir, "data1.wav"), output1, SR)
    sf.write(os.path.join(curr_writing_dir, "data2.wav"), output2, SR)


import pdb
pdb.set_trace()