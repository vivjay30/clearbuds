from conv_tasnet import ConvTasNet
import torch


model = ConvTasNet.load_model("checkpoints/clearvoice_2mics2voicesbg_12k_1pt5sec/final.pth.tar", input_channels=2)
dummy_input = torch.randn(1, 2, int(12000 * 1.5))
torch.onnx.export(model, dummy_input, '2channel_convtasnet_12k_1pt5.onnx', verbose=True, input_names=['encoder'], output_names=['decoder'])