import os
import glob
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


idx = 0
for algorithm in sorted(os.listdir(".")):
    if not os.path.isdir(algorithm): continue
    for dataset in sorted(os.listdir(algorithm)):
        if "hardware" in dataset: continue
        for experiment in sorted(os.listdir(os.path.join(algorithm, dataset))):
            file_path = os.path.join(algorithm, dataset, experiment)
            for file in sorted(os.listdir(file_path)):
                if "sdr" in file:
                    data = np.load(os.path.join(file_path, file))    
                    print("{} {} {} Median SI-SDRi {}".format(algorithm, dataset, experiment, np.median(data[:, 1] - data[:, 0])))
                elif "pesq" in file:
                    data = np.load(os.path.join(file_path, file))
                    print("{} {} {} Median Output PESQ {}".format(algorithm, dataset, experiment, np.median(data[:, 1])))
