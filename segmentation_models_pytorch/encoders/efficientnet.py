import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import *


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


efficientnet_encoders = {
    'efficientnet-b0': {
        'encoder': EfficientNet_3_Encoder,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 128),
        "params": {},
    },

    'efficientnet-b1': {
        'encoder': EfficientNet_3_Encoder,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 64),
        "params": {},
    },

    'efficientnet-b2': {
        'encoder': EfficientNet_3_Encoder,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 64),
        "params": {},
    },

    'efficientnet-b3': {
        'encoder': EfficientNet_3_Encoder,
        'pretrained_settings': None,
        'out_shapes': (1536, 136, 48, 32, 24),
        "params": {},
    },

    'efficientnet-b4': {
        'encoder': EfficientNet_3_Encoder,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 64),
        "params": {},
    },

    'efficientnet-b5': {
        'encoder': EfficientNet_5_Encoder,
        'pretrained_settings': None,
        'out_shapes': (2048, 176, 64, 40, 24),
        "params": {},
    },
}