import torch.nn as nn
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

import sys
sys.path.append("../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/")
from pretrainedmodels.models.torchvision_models import pretrained_settings
from .scse import SCse
from ..common.blocks import CBAM, CBAM_Module


class ResNetEncoder(ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.fc

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')
        super().load_state_dict(state_dict, **kwargs)


class ResNetEncoderSE(nn.Module):

    def __init__(self, encoder, attention_type="scse"):
        super().__init__()
        self.encode1 = nn.Sequential(encoder.conv1,
                                     encoder.bn1,
                                     encoder.relu)
        self.maxpool = encoder.maxpool
        if attention_type == "scse":
            self.encode2 = nn.Sequential(encoder.layer1,
                                         SCse(encoder.out_shapes[3]))
            self.encode3 = nn.Sequential(encoder.layer2,
                                         SCse(encoder.out_shapes[2]))
            self.encode4 = nn.Sequential(encoder.layer3,
                                         SCse(encoder.out_shapes[1]))
            self.encode5 = nn.Sequential(encoder.layer4,
                                         SCse(encoder.out_shapes[0]))
        elif attention_type == "cbam":
            self.encode2 = nn.Sequential(encoder.layer1,
                                         CBAM_Module(encoder.out_shapes[3]))
            self.encode3 = nn.Sequential(encoder.layer2,
                                         CBAM_Module(encoder.out_shapes[2]))
            self.encode4 = nn.Sequential(encoder.layer3,
                                         CBAM_Module(encoder.out_shapes[1]))
            self.encode5 = nn.Sequential(encoder.layer4,
                                         CBAM_Module(encoder.out_shapes[0]))

    def forward(self, x):
        x0 = self.encode1(x)

        x1 = self.maxpool(x0)
        x1 = self.encode2(x1)

        x2 = self.encode3(x1)
        x3 = self.encode4(x2)
        x4 = self.encode5(x3)

        return [x4, x3, x2, x1, x0]


resnet_encoders = {
    'resnet18': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet18'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [2, 2, 2, 2],
        },
    },

    'resnet34': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet34'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [3, 4, 6, 3],
        },
    },

    'resnet50': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet50'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
        },
    },

    'resnet101': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet101'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
        },
    },

    'resnet152': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet152'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 8, 36, 3],
        },
    },
}
