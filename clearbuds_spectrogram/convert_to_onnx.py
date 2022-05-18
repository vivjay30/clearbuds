import torch

from UNet import UNet

# Change to the path to calibration.pt
MODEL_PATH = "checkpoints/model_causal.pt"

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


# Load the UNet, sets the weights and params
# unet = UNet()
unet = torch.load(MODEL_PATH, map_location=device).to(device)
unet.eval()
unet.exporting = True

dummy_input = torch.randn(1, 1, 128, 64).to(device)

torch_out = unet(dummy_input)  # Not actualy used

# Export the model
torch.onnx.export(unet,               # model being run
                  dummy_input,                         # model input (or a tuple for multiple inputs)
                  "spectrogram.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  input_names = ['input'],   # the model's input names
                  output_names = ['mask']
                  # output_names = ['indices', 'heatmap'], # the model's output names},
                  # dynamic_axes = {'indices': {0: 'detections'}}
)
