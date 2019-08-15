from .decoder import UnetPPDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class UnetPP(EncoderDecoder):
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
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False,  # usefull for VGG models
            encoder_se_module=False,
            decoder_semodule=False,
            h_columns=False,
            deep_supervision=False,
            classification=False,
            linear_feature_unit=64
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights,
            se_module=encoder_se_module
        )

        decoder = UnetPPDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
            se_module=decoder_semodule,
            h_columns=h_columns,
            deep_supervision=deep_supervision,
            classification=classification,
            linear_feature_unit=linear_feature_unit
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)
