import torch
import coremltools as ct

from conv_tasnet import ConvTasNet
from cached_model import CachedModel

from thop import profile

RECEPTIVE_FIELD = 516
R = 2
X = 7
N = 256
H = 512


base_model = ConvTasNet(N=512, L=40, B=128, H=256, P=3, X=7, R=2, C=1)
base_model.eval()

dummy_input = torch.randn(1, 1, 15625 * 3)

torch.onnx.export(base_model,
                  dummy_input,
                  "ConvTasNet.onnx",
                  export_params=True,
                  input_names = ['input'],
                  output_names = ['output'])
