import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/")
from pretrainedmodels.models.dpn import DPN
from pretrainedmodels.models.dpn import pretrained_settings
from .scse import SCse
from ..common.blocks import CBAM, CBAM_Module


class DPNEncorder(DPN):

    def __init__(self, feature_blocks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_blocks = np.cumsum(feature_blocks)
        self.pretrained = False
        
        del self.last_linear

    def forward(self, x):

        features = []

        input_block = self.features[0]

        x = input_block.conv(x)
        x = input_block.bn(x)
        x = input_block.act(x)
        features.append(x)

        x = input_block.pool(x)

        for i, module in enumerate(self.features[1:], 1):
            x = module(x)
            if i in self.feature_blocks:
                features.append(x)

        out_features = [
            features[4],
            F.relu(torch.cat(features[3], dim=1), inplace=True),
            F.relu(torch.cat(features[2], dim=1), inplace=True),
            F.relu(torch.cat(features[1], dim=1), inplace=True),
            features[0],
        ]

        return out_features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)


class DPNEncorderSE(nn.Module):

    def __init__(self, encoder, attention_type="scse"):
        super().__init__()
        self.encoder = encoder
        self.feature_blocks = encoder.feature_blocks
        self.encode1 = nn.Sequential(encoder.features[0].conv,
                                    encoder.features[0].bn,
                                    encoder.features[0].act)
        self.encode2 = nn.Sequential(encoder.features[0].pool)
        if attention_type == "scse":
            self.se1 = SCse(encoder.out_shapes[3])
            self.se2 = SCse(encoder.out_shapes[2])
            self.se3 = SCse(encoder.out_shapes[1])
            self.se4 = SCse(encoder.out_shapes[0])
        elif attention_type == "cbam":
            self.se1 = CBAM_Module(encoder.out_shapes[3])
            self.se2 = CBAM_Module(encoder.out_shapes[2])
            self.se3 = CBAM_Module(encoder.out_shapes[1])
            self.se4 = CBAM_Module(encoder.out_shapes[0])

    def forward(self, x):

        features = []

        x = self.encode1(x)
        features.append(x)

        x = self.encode2(x)

        count = 1
        for i, module in enumerate(self.encoder.features[1:], 1):
            x = module(x)
            if i in self.feature_blocks:
                if count == 1:
                    x1 = torch.cat(x, dim=1)
                    x1 = self.se1(x1)
                    features.append(x1)
                if count == 2:
                    x2 = torch.cat(x, dim=1)
                    x2 = self.se2(x2)
                    features.append(x2)
                if count == 3:
                    x3 = torch.cat(x, dim=1)
                    x3 = self.se3(x3)
                    features.append(x3)
                if count == 4:
                    x4 = self.se4(x)
                    features.append(x4)
                count += 1

        out_features = [
            features[4],
            F.relu(features[3], inplace=True),
            F.relu(features[2], inplace=True),
            F.relu(features[1], inplace=True),
            features[0],
        ]
        return out_features


dpn_encoders = {
    'dpn68': {
        'encoder': DPNEncorder,
        'out_shapes': (832, 704, 320, 144, 10),
        'pretrained_settings': pretrained_settings['dpn68'],
        'params': {
            'feature_blocks': (3, 4, 12, 4),
            'groups': 32,
            'inc_sec': (16, 32, 32, 64),
            'k_r': 128,
            'k_sec': (3, 4, 12, 3),
            'num_classes': 1000,
            'num_init_features': 10,
            'small': True,
            'test_time_pool': True
        },
    },

    'dpn68b': {
        'encoder': DPNEncorder,
        'out_shapes': (832, 704, 320, 144, 10),
        'pretrained_settings': pretrained_settings['dpn68b'],
        'params': {
            'feature_blocks': (3, 4, 12, 4),
            'b': True,
            'groups': 32,
            'inc_sec': (16, 32, 32, 64),
            'k_r': 128,
            'k_sec': (3, 4, 12, 3),
            'num_classes': 1000,
            'num_init_features': 10,
            'small': True,
            'test_time_pool': True,
        },
    },

    'dpn92': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 1552, 704, 336, 64),
        'pretrained_settings': pretrained_settings['dpn92'],
        'params': {
            'feature_blocks': (3, 4, 20, 4),
            'groups': 32,
            'inc_sec': (16, 32, 24, 128),
            'k_r': 96,
            'k_sec': (3, 4, 20, 3),
            'num_classes': 1000,
            'num_init_features': 64,
            'test_time_pool': True
        },
    },

    'dpn98': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 1728, 768, 336, 96),
        'pretrained_settings': pretrained_settings['dpn98'],
        'params': {
            'feature_blocks': (3, 6, 20, 4),
            'groups': 40,
            'inc_sec': (16, 32, 32, 128),
            'k_r': 160,
            'k_sec': (3, 6, 20, 3),
            'num_classes': 1000,
            'num_init_features': 96,
            'test_time_pool': True,
        },
    },

    'dpn107': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 2432, 1152, 376, 128),
        'pretrained_settings': pretrained_settings['dpn107'],
        'params': {
            'feature_blocks': (4, 8, 20, 4),
            'groups': 50,
            'inc_sec': (20, 64, 64, 128),
            'k_r': 200,
            'k_sec': (4, 8, 20, 3),
            'num_classes': 1000,
            'num_init_features': 128,
            'test_time_pool': True
        },
    },

    'dpn131': {
        'encoder': DPNEncorder,
        'out_shapes': (2688, 1984, 832, 352, 128),
        'pretrained_settings': pretrained_settings['dpn131'],
        'params': {
            'feature_blocks': (4, 8, 28, 4),
            'groups': 40,
            'inc_sec': (16, 32, 32, 128),
            'k_r': 160,
            'k_sec': (4, 8, 28, 3),
            'num_classes': 1000,
            'num_init_features': 128,
            'test_time_pool': True
        },
    },

}
