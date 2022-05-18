# Instructures for ACM Badging

(Please download the source code and models here instead of cloning the repo: https://drive.google.com/file/d/1C1m7UjZRNVq84jzx49krJ5tsaaMwalRG/view?usp=sharing) 

## Overview
Because we have many results and ablation studies, we've included instructions to replicate the most significant result. That is the numerical results we report for our method in table 1, as well as the real demo video we show.

## Environment
Allthe results have been validated on a linux environment with a GPU. It shoudl work on a mac environment as well, but recently we have had signficant issues developing on the M1 chip. Older macbooks should be fine.

## Setup
First, make sure you are in the top level directory where this readme and 2 other subfolders are located. Add this folder to your pythonpath:
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

Next, pip install the dependencies in `requirements.txt`, for example `pip install -r requirements.txt`. Python 3.8 should work fine. The versions on the requirement version numbers are not super strict, but have been included for ease.

Then, unzip the audio data folder:
```
unzip 2voices_synthetic_test.zip
```



## Table 1 results
`cd` into `clearbuds_waveform/src/`

The following experiments will run our method (CB-Net) on the rendered data to produce the results in the top row of table 1.

Notice that there are three experiments in the table, which correspond to different combinations of background speakers and background noise. This first command is for BG+SV

If you don't have a GPU, you can use the flag `--use-cuda 0`

```
CUDA_VISIBLE_DEVICES=0 python evaluate_cascaded.py \
--model-path checkpoints/clearvoice_iphone_causal_mixed_l1spec_loss_large/final_37epochs.pth.tar \
--data-dir ../../test \
--n-mics 2 \
--n-speakers 2 \
--sample-rate 15625 \
--chunk-size 46850 --unet-checkpoint unet.pt
```

For sidevoice only (SV) add the flag `--no-background`
For background only (BG), remove the flag `--no-background` and change `--n-speakers` to 1 instead of 2.

## Real result used in the video
Run the following command (from the directory `clearbuds_waveform/src/`):

```
./cascaded.sh 20210909_16.30.53
```

If you don't have a gpu, you can use ```./cascaded_cpu.sh 20210909_16.30.53```


The output file is written as `real_output.wav` in the top level folder. The input is `real_input.wav`

This is the example used in our demo video: https://www.youtube.com/watch?v=d2y8dRSO-WE



