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

dummy_input = torch.randn(1, 1, 15625 * 3)

macs, params = profile(base_model, inputs=(dummy_input, ))

import pdb
pdb.set_trace()
traced_model = torch.jit.trace(base_model, (dummy_input))


# Encoder
ml_input1 = ct.TensorType(name='mixture', shape=(1, 1, 15625 * 3))

model = ct.convert(
    traced_model,
    inputs=[ml_input1], #name "input_1" is used in 'quickstart'
)
model.save("ConvTasNetFull.mlmodel")





