from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_unet(pretrained=False, **kwargs):
    """
    U-Net segmentation model with batch normalization for biomedical image segmentation
    pretrained (bool): load pretrained weights into the model
    in_channels (int): number of input channels
    out_channels (int): number of output channels
    init_features (int): number of feature-maps in the first encoder layer
    """
    model = CausalUNet(**kwargs)

    if pretrained:
        checkpoint = "https://github.com/mateuszbuda/brain-segmentation-pytorch/releases/download/v1.0/unet-e012d006.pt"
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=False, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    return model

class CausalUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=64, exporting=False):
        super(CausalUNet, self).__init__()

        features = init_features

        self.encoder1 = CausalUNet._block(in_channels, features, name="enc1_mod")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = CausalUNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = CausalUNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = CausalUNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = CausalUNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=(2, 1), stride=(2, 1)
        )
        self.decoder4 = CausalUNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=(2, 1), stride=(2, 1)
        )
        self.decoder3 = CausalUNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=(2, 1), stride=(2, 1)
        )
        self.decoder2 = CausalUNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=(2, 1), stride=(2, 1)
        )
        self.decoder1 = CausalUNet._block(features * 2, features, name="dec1")

        self.conv_mod = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        import pdb
        pdb.set_trace()
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))[:, :, :, 3:]

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4[:, :, :, 3:]), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3[:, :, :, 3:]), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2[:, :, :, 3:]), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1[:, :, :, 3:]), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv_mod(dec1))


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            groups=in_channels
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=in_channels)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=1,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


    def weighted_binary_cross_entropy(self, output, target, weights=None):        
        if weights is not None:
            assert len(weights) == 2
            
            loss = weights[1] * (target * torch.log(output + 1e-8)) + \
                   weights[0] * ((1 - target) * torch.log(1 - output + 1e-8))
        else:
            loss = target * torch.log(output + 1e-8) + (1 - target) * torch.log(1 - output + 1e-8)

        return torch.neg(torch.mean(loss))


    def load_pretrain(self, state_dict):  # pylint: disable=redefined-outer-name
        """Loads the pretrained keys in state_dict into model"""
        for key in state_dict.keys():
            try:
                _ = self.load_state_dict({key: state_dict[key]}, strict=False)
                print("Loaded {} (shape = {}) from the pretrained model".format(
                    key, state_dict[key].shape))
            except Exception as e:
                print("Failed to load {}".format(key))
                print(e)
