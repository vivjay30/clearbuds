import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_tasnet import ConvTasNet

L = 50
CHUNK = 400
JUMP = int(CHUNK / L)


class CachedModel(nn.Module):
    def __init__(self, path, input_channels=2, R=2, X=7, N=128, use_cuda=True):
        super(CachedModel, self).__init__()
        model = ConvTasNet.load_model(path, input_channels=input_channels)

        # model = ConvTasNet(N=256, L=50, B=256, H=512, P=3, X=X, R=2,
        #                    C=1, norm_type='cLN', causal=0,
        #                    mask_nonlinear='relu', input_channels=2)
        self.model = model
        self.N = N
        self.model.eval()
        if use_cuda:
            self.model.cuda()

        self.encoder_object = Encoder(model)
        self.layernorm_object = LayerNorm(model)
        self.separator_object = Separator(model)

        self.decoder_object = Decoder(model, self.N)

    # def forward(self):
    #     return
    def forward(self, mixture, encoder_buffer, layernorm_buffer):
        """
        mixture: 1 x 2 x CHUNK
        encoder_buffer: 1 x 128 x 516
        layernorm_buffer 1 x 128 x 516
        """
        encoder_buffer[:, :, :-JUMP] = encoder_buffer[:, :, JUMP:].clone()
        encoder_buffer[:, :, -JUMP:] = self.model.encoder(mixture)


        layernorm_buffer[:, :, :-JUMP] = layernorm_buffer[:, :, JUMP:].clone()
        layernorm_buffer[:, :, -JUMP:] = self.model.separator.network[0:2](encoder_buffer[:, :, -JUMP:])

        estimate_mask = self.model.separator.network[2:](layernorm_buffer)
        estimate_mask = F.relu(estimate_mask.view(1, 1, self.N, JUMP))

        estimated_source = self.model.decoder(encoder_buffer, estimate_mask)

        # estimate_source = mixture
        return [estimated_source, encoder_buffer, layernorm_buffer]


    # def encoder(self, mixture, encoder_buffer):
    #     encoder_buffer[:, :, :-14] = encoder_buffer[:, :, 14:].clone()
    #     encoder_buffer[:, :, -14:] = self.model.encoder(mixture)

    #     return encoder_buffer

    # def layernorm(self, encoder_buffer, layernorm_buffer):
    #     layernorm_buffer[:, :, :-14] = layernorm_buffer[:, :, 14:].clone()
    #     layernorm_buffer[:, :, -14:] = self.model.separator.network[0:2](encoder_buffer[:, :, -14:])

    #     return layernorm_buffer

    # def separator(self, layernorm_buffer):
    #     estimate_mask = self.model.separator.network[2:](layernorm_buffer)
    #     estimate_mask = F.relu(estimate_mask.view(1, 1, 128, 14))
    #     return estimate_mask

    # def decoder(self, encoder_buffer, estimate_mask):
    #     estimated_source = self.model.decoder(encoder_buffer, estimate_mask)
    #     return estimated_source

class Encoder(nn.Module):
    def __init__(self, model):
        super(Encoder, self).__init__()
        self.model = model

    def forward(self, mixture, encoder_buffer):
        #tmp = encoder_buffer[:, :, 14:]
        #encoder_buffer[:, :, :-14] = tmp
        data = self.model.encoder(mixture)
        return torch.cat((encoder_buffer[:, :, JUMP:], data), 2)

        # return self.model.encoder(mixture)


class LayerNorm(nn.Module):
    def __init__(self, model):
        super(LayerNorm, self).__init__()
        self.model = model
        self.conv_layers = []
        R = 2
        X = 7
        for r in range(R):
            for x in range(X):
                self.conv_layers.append(TemporalConvBlock(self.model.separator.network[2][r][x]))


    def forward(self, encoder_buffer, conv_buffers):
        """
        Encoder buffer: B x 128 x 522
        Conv Buffers: 15 x B x 128 x 522
        """
        R = 2
        X = 7
        # Layernorm
        new_conv_buffers = torch.cat((conv_buffers[0, :, :, JUMP:], self.model.separator.network[0:2](encoder_buffer[:, :, -JUMP:])), 2).unsqueeze(0)
        
        # 1D Convolutions
        for i in range(0, R * X):
            output = self.conv_layers[i](new_conv_buffers[i], conv_buffers[i+1]).unsqueeze(0)
            new_conv_buffers = torch.cat((new_conv_buffers, output), 0)
            #conv_buffers[i+1] = self.conv_layers[i](conv_buffers[i], conv_buffers[i+1])

        return new_conv_buffers


class Separator(nn.Module):
    def __init__(self, model):
        super(Separator, self).__init__()
        self.model = model

    def forward(self, layernorm_buffer):
        estimate_mask = self.model.separator.network[2:](layernorm_buffer)
        return estimate_mask


class TemporalConvBlock(nn.Module):
    def __init__(self, block):
        super(TemporalConvBlock, self).__init__()
        self.block = block
        # Yes this is ugly
        self.dilation = block.net[3].net[0].dilation[0]
        self.K = block.net[3].net[0].kernel_size[0]
        self.receptive_field = self.K + 2 * (self.dilation - 1)


    def forward(self, input_buffer, output_buffer):
        # How many input samples we need to get 14 output samples
        total_receptive_field = self.receptive_field + (JUMP - 1)
        output = self.block(input_buffer[:, :, -total_receptive_field:])
        assert(output.shape[-1] == JUMP)

        return torch.cat((output_buffer[:, :, JUMP:], output), 2)


class Decoder(nn.Module):
    def __init__(self, model, N=128):
        super(Decoder, self).__init__()
        self.model = model
        self.N = N

    def forward(self, encoder_buffer, estimate_mask):
        # Do the last pass from the separator here
        estimate_mask = self.model.separator.network[3](estimate_mask[-1, :, :, -JUMP:])
        return self.model.decoder(encoder_buffer, F.relu(estimate_mask.view(1, 1, self.N, JUMP)))



