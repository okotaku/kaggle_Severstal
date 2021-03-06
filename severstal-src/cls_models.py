import sys
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torchvision import models
#from efficientnet_pytorch import EfficientNet

import sys
sys.path.append("../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/")
from pretrainedmodels import se_resnext101_32x4d, senet154
from pretrainedmodels import inceptionresnetv2
from pretrainedmodels import pnasnet5large
from pretrainedmodels import xception


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features=5004):
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()
        # nn.init.kaiming_uniform_(self.weight)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        # x=self.head(features)
        # print(self.weight.shape)
        # self.fc1.weight=nn.Parameter(F.normalize(self.fc1.weight)).cuda()
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        # cosine = cosine.clamp(-1, 1)
        # self.fc1(F.normalize(x))
        # F.linear(F.normalize(x), F.normalize(self.weight.cuda()))
        return cosine


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class MaxPool(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, x.shape[2:])


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class CSE(nn.Module):
    def __init__(self, in_ch, r):
        super(CSE, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]),-1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.sigmoid(x)

        x = input_x * x

        return x


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x


class SCSE(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)

        x = cSE + sSE

        return x


class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x

        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def create_net(net_cls, pretrained: bool):
    #net = net_cls()
    #if pretrained is not None:
    #    net.load_state_dict(torch.load(pretrained))
    net = net_cls(pretrained=pretrained)
    return net


class ResNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=models.resnet50):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AdaptiveConcatPool2d()
        self.net.fc = nn.Sequential(
            Flatten(),
            #SEBlock(2048*2),
            nn.Dropout(),
            nn.Linear(2048*2, num_classes)
        )

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=models.densenet121):
        super().__init__()
        self.net = net_cls(pretrained=pretrained)
        self.avg_pool = AdaptiveConcatPool2d()
        self.net.classifier = nn.Sequential(
            Flatten(),
            SEBlock(1024*2),
            nn.Dropout(),
            nn.Linear(1024*2, num_classes)
        )

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out)
        out = self.net.classifier(out)
        return out


class SEResNext(nn.Module):
    def __init__(self, num_classes, pretrained="imagenet", pool_type="concat"):
        super().__init__()
        self.net = se_resnext101_32x4d(pretrained=pretrained)
        if pool_type == "concat":
            self.net.avg_pool = AdaptiveConcatPool2d()
            last_channel = 2048*2
        elif pool_type == "gem":
            self.net.avg_pool = GeM()
            last_channel = 2048
        self.net.last_linear = nn.Sequential(
            Flatten(),
            SEBlock(last_channel),
            nn.Dropout(),
            nn.Linear(last_channel, num_classes)
        )


    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class Efficient(nn.Module):
    def __init__(self, num_classes, encoder='efficientnet-b7'):
        super().__init__()
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        self.net = EfficientNet.from_pretrained(encoder)
        self.avg_pool = AdaptiveConcatPool2d()
        self.classifier = nn.Sequential(
            Flatten(),
            SEBlock(n_channels_dict[encoder] * 2),
            nn.Dropout(),
            nn.Linear(n_channels_dict[encoder] * 2, num_classes)
        )

    def forward(self, x):
        x = self.net.extract_features(x)
        x = self.avg_pool(x)
        out = self.classifier(x)

        return out


class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes, pretrained="imagenet"):
        super().__init__()
        self.net = inceptionresnetv2(pretrained=pretrained)
        self.net.avgpool_1a = AdaptiveConcatPool2d()
        self.net.last_linear = nn.Sequential(
            Flatten(),
            SEBlock(1536*2),
            nn.Linear(1536*2, num_classes)
        )


    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class SENet(nn.Module):
    def __init__(self, num_classes, pretrained="senet154-c7b49a05.pth", dropout=False, arcface=False):
        super().__init__()
        self.net = senet154(pretrained=pretrained)
        self.net.last_linear = nn.Linear(2048, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class SENetV2(nn.Module):
    def __init__(self, num_classes, pretrained="senet154-c7b49a05.pth", dropout=False, arcface=False):
        super().__init__()
        self.net = senet154(pretrained=pretrained)
        self.net.avg_pool = AvgPool()
        self.net.last_linear = nn.Linear(2048, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class SENetV3(nn.Module):
    def __init__(self, num_classes, pretrained="senet154-c7b49a05.pth", dropout=False, arcface=False):
        super().__init__()
        self.net = senet154(pretrained=pretrained)
        self.net.avg_pool = AvgPool()
        self.net.last_linear = nn.Linear(2048, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class SENetV4(nn.Module):
    def __init__(self, num_classes, pretrained="senet154-c7b49a05.pth", dropout=False, arcface=False):
        super().__init__()
        self.net = senet154(pretrained=pretrained)
        self.net.avg_pool = AdaptiveConcatPool2d()
        self.net.last_linear = nn.Sequential(
            Flatten(),
            SEBlock(2048*2),
            nn.Linear(2048*2, num_classes)
        )

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class PNASNet5Large(nn.Module):
    def __init__(self, num_classes, pretrained="pnasnet5large-bf079911.pth", dropout=False, arcface=False):
        super().__init__()
        self.net = pnasnet5large(pretrained=pretrained)
        self.net.avg_pool = AdaptiveConcatPool2d()
        self.net.last_linear = nn.Sequential(
            Flatten(),
            SEBlock(4320 * 2),
            nn.Linear(4320 * 2, num_classes)
        )

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)

class Xception(nn.Module):
    def __init__(self, num_classes, pretrained="xception-43020ad28.pth"):
        super().__init__()
        self.net = xception(pretrained=pretrained)
        self.avg_pool = AdaptiveConcatPool2d()
        self.net.last_linear = nn.Sequential(
            Flatten(),
            SEBlock(2048 * 2),
            nn.Linear(2048 * 2, num_classes)
        )

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out)
        out = self.net.last_linear(out)
        return out