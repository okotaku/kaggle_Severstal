import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import *
from ..unet.decoder import UnetDecoder


class EfficientNet_5_Encoder(nn.Module):

    def __init__(self):
        super(EfficientNet_5_Encoder, self).__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b5')

    def forward(self, inputs):
        x = relu_fn(self.model._bn0(self.model._conv_stem(inputs)))

        global_features = []

        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in [2, 7, 12, 26]:
                global_features.append(x)
        x = relu_fn(self.model._bn1(self.model._conv_head(x)))
        global_features.append(x)
        global_features.reverse()

        return global_features


class EfficientNet_5_unet(nn.Module):
    def __init__(
            self,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            center=False,  # usefull for VGG models
            encoder_se_module=False,
            decoder_semodule=False,
            h_columns=False,
            act="relu",
            skip=False,
            use_transpose=False,
            freeze_bn=False,
            freeze_bn_affine=False,
            classification=False,
            attention_type="scse"
    ):
        super(EfficientNet_5_unet, self).__init__()
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine

        self.model_encoder = EfficientNet_5_Encoder()
        self.model_decoder = UnetDecoder(
            encoder_channels=(2048, 176, 64, 40, 24),
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
            se_module=decoder_semodule,
            h_columns=h_columns,
            act=act,
            skip=skip,
            use_transpose=use_transpose,
            classification=classification,
            attention_type=attention_type
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(nn.Linear(2048, 2048, bias=True), nn.Linear(2048, 1, bias=True))
        self.fea_bn = nn.BatchNorm1d(512)
        self.fea_bn.bias.requires_grad_(False)

    def forward(self, x):

        global_features = self.model_encoder(x)
        seg_feature = self.model_decoder(global_features)

        return seg_feature

    def train(self, mode=True):
        super(EfficientNet_5_unet, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False


class EfficientNet_3_Encoder(nn.Module):

    def __init__(self):
        super(EfficientNet_3_Encoder, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3')

    def forward(self, inputs):
        x = relu_fn(self.model._bn0(self.model._conv_stem(inputs)))

        global_features = []

        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in [1, 4, 7, 17]:
                global_features.append(x)
        x = relu_fn(self.model._bn1(self.model._conv_head(x)))
        global_features.append(x)
        global_features.reverse()

        return global_features


class EfficientNet_3_unet(nn.Module):
    def __init__(
            self,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            center=False,  # usefull for VGG models
            encoder_se_module=False,
            decoder_semodule=False,
            h_columns=False,
            act="relu",
            skip=False,
            use_transpose=False,
            freeze_bn=False,
            freeze_bn_affine=False,
            classification=False,
            attention_type="scse"
    ):
        super(EfficientNet_3_unet, self).__init__()
        self.model_encoder = EfficientNet_3_Encoder()
        self.model_decoder = UnetDecoder(
            encoder_channels=(1536, 136, 48, 32, 24),
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
            se_module=decoder_semodule,
            h_columns=h_columns,
            act=act,
            skip=skip,
            use_transpose=use_transpose,
            classification=classification,
            attention_type=attention_type
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(nn.Linear(2048, 2048, bias=True), nn.Linear(2048, 1, bias=True))
        self.fea_bn = nn.BatchNorm1d(512)
        self.fea_bn.bias.requires_grad_(False)

    def forward(self, x):

        global_features = self.model_encoder(x)
        seg_feature = self.model_decoder(global_features)

        return seg_feature

    def train(self, mode=True):
        super(EfficientNet_3_unet, self).train(mode)
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if self.freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
