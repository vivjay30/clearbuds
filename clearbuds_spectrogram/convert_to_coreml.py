import cv2
import torch
import coremltools as ct

import numpy as np

from UNet import UNet
from coremltools.models.neural_network import flexible_shape_utils

MODEL_PATH = "checkpoints/model.pt"

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# unet = UNet()
unet = torch.load(MODEL_PATH, map_location=device).to(device)
unet.eval()
unet.exporting = True

dummy_input = torch.randn(1, 1, 128, 64).to(device)
traced_model = torch.jit.trace(unet, (dummy_input))

# Multiples of 32 only
ml_input = ct.TensorType(name="sample_input", shape=(1, 1, 128, 64))

# Convert to CoreML
model = ct.convert(
    traced_model,
    inputs=[ml_input],
)

model.save("spectrogram.mlmodel")

# Flexible input size
# spec = ct.utils.load_spec("spectrogram.mlmodel")
# nn = spec.neuralNetwork

# img_size_ranges = flexible_shape_utils.NeuralNetworkImageSizeRange()
# img_size_ranges.add_width_range((64, 9984))
# img_size_ranges.add_height_range((128, 128))
# flexible_shape_utils.update_image_size_range(spec, feature_name="sample_input",
#                                              size_range=img_size_ranges)
# ct.models.utils.save_spec(spec, "spectrogram.mlmodel")
