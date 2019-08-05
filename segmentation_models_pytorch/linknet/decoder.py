import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv2dReLU
from ..base.model import Model
from ..encoders.scse import SCse


class TransposeX2(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True, **batchnorm_params):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, **batchnorm_params))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, se_module=False):
        super().__init__()
        if se_module:
            self.block = nn.Sequential(
                Conv2dReLU(in_channels, in_channels // 4, kernel_size=1, use_batchnorm=use_batchnorm),
                TransposeX2(in_channels // 4, in_channels // 4, use_batchnorm=use_batchnorm),
                Conv2dReLU(in_channels // 4, out_channels, kernel_size=1, use_batchnorm=use_batchnorm),
                SCse(out_channels)
            )
        else:
            self.block = nn.Sequential(
                Conv2dReLU(in_channels, in_channels // 4, kernel_size=1, use_batchnorm=use_batchnorm),
                TransposeX2(in_channels // 4, in_channels // 4, use_batchnorm=use_batchnorm),
                Conv2dReLU(in_channels // 4, out_channels, kernel_size=1, use_batchnorm=use_batchnorm),
            )

    def forward(self, x):
        x, skip = x
        x = self.block(x)
        if skip is not None:
            x = x + skip
        return x


class LinknetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            prefinal_channels=32,
            final_channels=1,
            use_batchnorm=True,
            se_module=False,
            h_columns=False
    ):
        super().__init__()

        in_channels = encoder_channels
        self.h_columns = h_columns

        self.layer1 = DecoderBlock(in_channels[0], in_channels[1], use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer2 = DecoderBlock(in_channels[1], in_channels[2], use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer3 = DecoderBlock(in_channels[2], in_channels[3], use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer4 = DecoderBlock(in_channels[3], in_channels[4], use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer5 = DecoderBlock(in_channels[4], prefinal_channels, use_batchnorm=use_batchnorm, se_module=se_module)

        if self.h_columns:
            self.layer1_h = nn.Conv2d(in_channels[1], prefinal_channels, kernel_size=(1, 1))
            self.layer2_h = nn.Conv2d(in_channels[2], prefinal_channels, kernel_size=(1, 1))
            self.layer3_h = nn.Conv2d(in_channels[3], prefinal_channels, kernel_size=(1, 1))
            self.layer4_h = nn.Conv2d(in_channels[4], prefinal_channels, kernel_size=(1, 1))
            self.final_conv = nn.Sequential(nn.Conv2d(int(prefinal_channels * 5), 64, kernel_size=3, padding=1),
                                            nn.ELU(True),
                                            nn.Conv2d(64, 1, kernel_size=1, bias=False))
        else:
            self.final_conv = nn.Conv2d(prefinal_channels, final_channels, kernel_size=(1, 1))

        self.initialize()

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.h_columns:
            d5 = self.layer1([encoder_head, skips[0]])
            d4 = self.layer2([d5, skips[1]])
            d3 = self.layer3([d4, skips[2]])
            d2 = self.layer4([d3, skips[3]])
            d1 = self.layer5([d2, None])
            x = torch.cat((d1,
                           self.layer4_h(F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)),
                           self.layer3_h(F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True)),
                           self.layer2_h(F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True)),
                           self.layer1_h(F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True))), 1)
        else:
            x = self.layer1([encoder_head, skips[0]])
            x = self.layer2([x, skips[1]])
            x = self.layer3([x, skips[2]])
            x = self.layer4([x, skips[3]])
            x = self.layer5([x, None])
        x = self.final_conv(x)

        return x
