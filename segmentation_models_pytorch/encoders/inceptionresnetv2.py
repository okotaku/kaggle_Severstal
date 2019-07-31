import torch.nn as nn

import sys
sys.path.append("../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/")
from pretrainedmodels.models.inceptionresnetv2 import InceptionResNetV2
from pretrainedmodels.models.inceptionresnetv2 import pretrained_settings
from .scse import SCse


class InceptionResNetV2Encoder(InceptionResNetV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pretrained = False

        # correct paddings
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

        # remove linear layers
        del self.avgpool_1a
        del self.last_linear

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x0 = x

        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x1 = x

        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x2 = x

        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x3 = x

        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        x4 = x

        features = [x4, x3, x2, x1, x0]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)


class InceptionResNetV2SE(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        self.encode1 = nn.Sequential(encoder.conv2d_1a,
                                    encoder.conv2d_2a,
                                    encoder.conv2d_2b)
        self.encode2 = nn.Sequential(encoder.maxpool_3a,
                                     encoder.conv2d_3b,
                                     encoder.conv2d_4a,
                                     SCse(encoder.out_shapes[3]))
        self.encode3 = nn.Sequential(encoder.maxpool_5a,
                                     encoder.mixed_5b,
                                     encoder.repeat,
                                     SCse(encoder.out_shapes[2]))
        self.encode4 = nn.Sequential(encoder.mixed_6a,
                                     encoder.repeat_1,
                                     SCse(encoder.out_shapes[1]))
        self.encode5 = nn.Sequential(encoder.mixed_7a,
                                     encoder.repeat_2,
                                     encoder.block8,
                                     encoder.conv2d_7b,
                                     SCse(encoder.out_shapes[0]))

    def forward(self, x):
        x0 = self.encode1(x)
        x1 = self.encode2(x0)
        x2 = self.encode3(x1)
        x3 = self.encode4(x2)
        x4 = self.encode5(x3)

        return [x4, x3, x2, x1, x0]


inception_encoders = {
    'inceptionresnetv2': {
        'encoder': InceptionResNetV2Encoder,
        'pretrained_settings': pretrained_settings['inceptionresnetv2'],
        'out_shapes': (1536, 1088, 320, 192, 64),
        'params': {
            'num_classes': 1000,
        }

    }
}
