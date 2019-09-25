import functools
import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders, ResNetEncoderSE
from .dpn import dpn_encoders, DPNEncorderSE
from .vgg import vgg_encoders
from .senet import senet_encoders, SENetEncoderSE
from .densenet import densenet_encoders, DenseNetSE
from .inceptionresnetv2 import inception_encoders, InceptionResNetV2SE
#from .efficient import efficientnet_encoders
from .scse import SCse

from ._preprocessing import preprocess_input

encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inception_encoders)
#encoders.update(efficientnet_encoders)


def get_encoder(name, encoder_weights=None, se_module=False, attention_type="scse"):
    Encoder = encoders[name]['encoder']
    encoder = Encoder(**encoders[name]['params'])
    encoder.out_shapes = encoders[name]['out_shapes']

    if encoder_weights is not None:
        settings = encoders[name]['pretrained_settings'][encoder_weights]
        encoder.load_state_dict(model_zoo.load_url(settings['url']))
    if se_module:
        if name in resnet_encoders.keys():
            encoder = ResNetEncoderSE(encoder, attention_type=attention_type)
            encoder.out_shapes = encoders[name]['out_shapes']
        elif name in senet_encoders.keys():
            encoder = SENetEncoderSE(encoder, attention_type=attention_type)
            encoder.out_shapes = encoders[name]['out_shapes']
        elif name in inception_encoders.keys():
            encoder = InceptionResNetV2SE(encoder, attention_type=attention_type)
            encoder.out_shapes = encoders[name]['out_shapes']
        elif name in densenet_encoders.keys():
            encoder = DenseNetSE(encoder, attention_type=attention_type)
            encoder.out_shapes = encoders[name]['out_shapes']
        elif name in dpn_encoders.keys():
            encoder = DPNEncorderSE(encoder, attention_type=attention_type)
            encoder.out_shapes = encoders[name]['out_shapes']

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_fn(encoder_name, pretrained='imagenet'):
    settings = encoders[encoder_name]['pretrained_settings']

    if pretrained not in settings.keys():
        raise ValueError('Avaliable pretrained options {}'.format(settings.keys()))

    input_space = settings[pretrained].get('input_space')
    input_range = settings[pretrained].get('input_range')
    mean = settings[pretrained].get('mean')
    std = settings[pretrained].get('std')
    
    return functools.partial(preprocess_input, mean=mean, std=std, input_space=input_space, input_range=input_range)
