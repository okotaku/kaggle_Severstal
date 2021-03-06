import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import PANetDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class PANet(EncoderDecoder):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        center: if ``True`` add ``Conv2dReLU`` block on encoder head (useful for VGG models)

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(64, 64, 64, 64, 64),
            classes=1,
            activation='sigmoid',
            center=True,  # usefull for VGG models
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
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights,
            se_module=encoder_se_module,
            attention_type=attention_type
        )

        decoder = PANetDecoder(
            encoder_channels=encoder.out_shapes,
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

        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)

    def train(self, mode=True):
        super(PANet, self).train(mode)
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
