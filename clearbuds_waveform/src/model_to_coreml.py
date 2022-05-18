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


base_model = CachedModel("checkpoints/clearvoice_iphone_causal_anechoic_reverb/final.pth.tar", input_channels=2, N=N, use_cuda=False)

total_macs = 0
# Encoder
dummy_input = torch.randn(1, 2, 400)
encoder_buffer = torch.randn(1, N, RECEPTIVE_FIELD)

# macs, params = profile(base_model.encoder_object, inputs=(dummy_input, encoder_buffer))
# total_macs += params

traced_model = torch.jit.trace(base_model.encoder_object, (dummy_input, encoder_buffer))

ml_input1 = ct.TensorType(name='mixture', shape=(1, 2, 400))
ml_input2 = ct.TensorType(name='encoder_buffer', shape=(1, N, RECEPTIVE_FIELD))

model = ct.convert(
    traced_model,
    inputs=[ml_input1, ml_input2], #name "input_1" is used in 'quickstart'
)
model.save("ClearVoiceEncoder.mlmodel")


# LayerNorm, Bottleneck Conv
dummy_input1 = torch.randn(1, N, RECEPTIVE_FIELD)
dummy_input2 = torch.randn(15, 1, N, RECEPTIVE_FIELD)

# macs, params = profile(base_model.layernorm_object, inputs=(dummy_input1, dummy_input2))
# total_macs += params

traced_model = torch.jit.trace(base_model.layernorm_object, (dummy_input1, dummy_input2))
ml_input1 = ct.TensorType(name='encoder_buffer', shape=(1, N, RECEPTIVE_FIELD))
ml_input2 = ct.TensorType(name='conv_buffers', shape=(15, 1, N, RECEPTIVE_FIELD))

model = ct.convert(
    traced_model,
    inputs=[ml_input1, ml_input2], #name "input_1" is used in 'quickstart'
)
model.save("ClearVoiceLayerNorm.mlmodel")

# # Separator
# dummy_input = torch.randn(1, 128, RECEPTIVE_FIELD)
# traced_model = torch.jit.trace(base_model.separator_object, dummy_input)
# ml_input = ct.TensorType(name='layernorm_buffer', shape=(1, 128, RECEPTIVE_FIELD))

# model = ct.convert(
#     traced_model,
#     inputs=[ml_input], #name "input_1" is used in 'quickstart'
# )

# model.save("ClearVoiceSeparator.mlmodel")


# Decoder
dummy_input1 = torch.randn(1, N, RECEPTIVE_FIELD)
dummy_input2 = torch.randn(15, 1, N, RECEPTIVE_FIELD)


traced_model = torch.jit.trace(base_model.decoder_object, (dummy_input1, dummy_input2))

# macs, params = profile(base_model.decoder_object, inputs=(dummy_input1, dummy_input2))
# total_macs += params

ml_input1 = ct.TensorType(name='encoder_buffer', shape=(1, N, RECEPTIVE_FIELD))
ml_input2 = ct.TensorType(name='conv_buffers', shape=(15, 1, N, RECEPTIVE_FIELD))

model = ct.convert(
    traced_model,
    inputs=[ml_input1, ml_input2], #name "input_1" is used in 'quickstart'
)

# print("Total Macs: {}".format(total_macs))

model.save("ClearVoiceDecoder.mlmodel")





