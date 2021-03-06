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

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skips is not None:
            for skip in skips:
                x += skip
        x = self.block(x)
        return x


class CenterBlock(DecoderBlock):

    def forward(self, x):
        return self.block(x)


class FPAv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.upsample(x_glob, scale_factor=16, mode='bilinear', align_corners=True)  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        d3 = F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2

        x = x + x_glob

        return x


class LinknetPPDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            se_module=False,
            h_columns=False,
            deep_supervision=False
    ):
        super().__init__()
        self.h_columns = h_columns
        self.deep_supervision = deep_supervision

        if center:
            channels = encoder_channels[0]
            #self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
            self.center = FPAv2(channels, channels)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer4_1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer3_1 = DecoderBlock(encoder_channels[1] + encoder_channels[2], out_channels[1],
                                       use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer3_2 = DecoderBlock(in_channels[1] + out_channels[1], out_channels[1],
                                       use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer2_1 = DecoderBlock(encoder_channels[2] + encoder_channels[3], out_channels[2],
                                       use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer2_2 = DecoderBlock(in_channels[2] + out_channels[2], out_channels[2],
                                       use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer2_3 = DecoderBlock(in_channels[2] + out_channels[2] * 2, out_channels[2],
                                       use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer1_1 = DecoderBlock(encoder_channels[3] + encoder_channels[4], out_channels[3],
                                       use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer1_2 = DecoderBlock(in_channels[3] + out_channels[3], out_channels[3],
                                       use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer1_3 = DecoderBlock(in_channels[3] + out_channels[3], out_channels[3],
                                       use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer1_4 = DecoderBlock(in_channels[3] + out_channels[3] * 3, out_channels[3],
                                       use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer0 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer0_1 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer0_2 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm, se_module=se_module)
        self.layer0_3 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm, se_module=se_module)

        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))
        self.final_conv1 = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))
        self.final_conv2 = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))
        self.final_conv3 = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        d5 = self.layer4_1(encoder_head, [skips[0]])

        d4_1 = self.layer3_1(skips[0], [skips[1]])
        d4 = self.layer3_2(d5, [skips[1], d4_1])

        d3_1 = self.layer2_1(skips[1], [skips[2]])
        d3_2 = self.layer2_2(d4_1, [skips[2], d3_1])
        d3 = self.layer2_3(d4, [skips[2], d3_1, d3_2])

        d2_1 = self.layer1_1(skips[2], [skips[3]])
        d2_2 = self.layer1_2(d3_1, [skips[3], d2_1])
        d2_3 = self.layer1_3(d3_2, [skips[3], d2_2])
        d2 = self.layer1_4(d3, [skips[3], d2_1, d2_2, d2_3])

        d1 = self.layer0(d2, None)
        x = self.final_conv(d1)
        if self.deep_supervision:
            d1_1 = self.layer0_1(d2_1, None)
            d1_2 = self.layer0_2(d2_2, None)
            d1_3 = self.layer0_3(d2_3, None)

            x1 = self.final_conv1(d1_1)
            x2 = self.final_conv2(d1_2)
            x3 = self.final_conv3(d1_3)
            return [x, x1, x2, x3]

        return x
