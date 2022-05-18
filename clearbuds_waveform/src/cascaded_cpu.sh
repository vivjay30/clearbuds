FILE_L=real_examples/${1}_L.wav
FILE_R=real_examples/${1}_R.wav

python evaluate_recorded.py --model-path checkpoints/clearvoice_iphone_causal_anechoic_reverb/final.pth.tar \
											        --n-mics 2 --sample-rate 15625 --file-path-left $FILE_L --use-cuda=0;
mv evaluation_outputs/output.wav evaluation_outputs/Conv-TasNet-Only.wav;
cp evaluation_outputs/Conv-TasNet-Only.wav ../../clearbuds_spectrogram/inputs/output0.wav;
cp $FILE_L ../../clearbuds_spectrogram/inputs/mic00_voice00.wav;
cp $FILE_R ../../clearbuds_spectrogram/inputs/mic01_voice00.wav;
cd ../../clearbuds_spectrogram/;

python inference_causal.py inputs checkpoints/model.pt outputs  --sample-rate 15625 --chunk-size 25600 --cutoff .003;
mv outputs/output.wav ../real_output.wav;

cd ../clearbuds_waveform;

# For doing spectrogram only
# CUDA_VISIBLE_DEVICES=2 python inference_causal.py inputs checkpoints/model.pt outputs --sample-rate 15625 --chunk-size 25600 --cutoff .006 --spectrogram-only;
# mv outputs/output.wav outputs/spectrogram.wav;