import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, act="relu", **batchnorm_params):

        super().__init__()

        if act == "elu":
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride=stride, padding=padding, bias=not (use_batchnorm)),
                nn.ELU(True),
            ]
        elif act == "swish":
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride=stride, padding=padding, bias=not (use_batchnorm)),
                Swish(),
            ]
        else:
            layers = [
              nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, bias=not (use_batchnorm)),
              nn.ReLU(inplace=True),
            ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

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

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp1 = _ASPPModule(inplanes, mid_c, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(mid_c),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(mid_c * 5, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return sinusoid_table

def get_sinusoid_encoding_table_2d(H,W, d_hid):
    ''' Sinusoid position encoding table '''
    n_position=H*W
    sinusoid_table=get_sinusoid_encoding_table(n_position,d_hid)
    sinusoid_table=sinusoid_table.reshape(H,W,d_hid)
    return sinusoid_table


class CBAM_Module(nn.Module):
    def __init__(self, channels, reduction=4,attention_kernel_size=3,position_encode=False):
        super(CBAM_Module, self).__init__()
        self.position_encode=position_encode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        if self.position_encode:
            k=3
        else:
            k=2
        self.conv_after_concat = nn.Conv2d(k, 1,
                                           kernel_size = attention_kernel_size,
                                           stride=1,
                                           padding = attention_kernel_size//2)
        self.sigmoid_spatial = nn.Sigmoid()
        self.position_encoded=None

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        b, c, h, w = x.size()
        if self.position_encode:
            if self.position_encoded is None:

                pos_enc=get_sinusoid_encoding_table(h,w)
                pos_enc=Variable(torch.FloatTensor(pos_enc),requires_grad=False)
                if x.is_cuda:
                    pos_enc=pos_enc.cuda()
                self.position_encoded=pos_enc
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        if self.position_encode:
            pos_enc=self.position_encoded
            pos_enc = pos_enc.view(1, 1, h, w).repeat(b, 1, 1, 1)
            x = torch.cat((avg, mx,pos_enc), 1)
        else:
            x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


#https://github.com/arc144/siim-pneumothorax/blob/master/Models/layers.py
def Norm2d(planes):
    return nn.BatchNorm2d(planes)


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), groups=1, dilation=1, act=False):
        super(ConvBn2d, self).__init__()
        self.act = act
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=True,
                              groups=groups,
                              dilation=dilation)
        self.bn = Norm2d(out_channels)
        if self.act:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x


class GlobalAttentionUpsample(nn.Module):
    def __init__(self, skip_channels, channels, out_channels=None):
        super(GlobalAttentionUpsample, self).__init__()
        self.out_channels = out_channels
        if out_channels is None:
            out_channels = channels
        self.conv3 = nn.Conv2d(skip_channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConvBn2d(channels, channels, kernel_size=1, padding=0)
        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if out_channels is not None:
            self.conv_out = ConvBn2d(channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip, up=True):
        # Reduce channels
        skip = self.conv3(skip)
        # Upsample
        if up:
            x = self.upsample(x)
        # GlobalPool and conv1
        cal1 = self.GPool(x)
        cal1 = self.conv1(cal1)
        cal1 = self.relu(cal1)

        # Calibrate skip connection
        skip = cal1 * skip
        # Add
        x = x + skip
        if self.out_channels is not None:
            x = self.conv_out(x)
        return x


###########################################################################
############################ PANet BLOCKS #################################
###########################################################################
class FeaturePyramidAttention(nn.Module):
    def __init__(self, channels, out_channels=None):
        super(FeaturePyramidAttention, self).__init__()
        if out_channels is None:
            out_channels = channels
        self.conv1 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1p = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv3a = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv5a = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv5b = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

        self.conv7a = nn.Conv2d(channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.conv7b = nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3)

        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, mode='std'):
        H, W = x.shape[2:]
        # down-path
        if mode == 'std':
            xup1 = self.downsample(x)
            xup1 = self.conv7a(xup1)
        elif mode == 'reduced':
            xup1 = self.conv7a(x)
        elif mode == 'extended':
            xup1 = F.avg_pool2d(x, kernel_size=4, stride=4)
            xup1 = self.conv7a(xup1)

        xup2 = self.downsample(xup1)
        xup2 = self.conv5a(xup2)

        xup3 = self.downsample(xup2)
        xup3 = self.conv3a(xup3)

        # Skips
        x1 = self.conv1(x)
        xup1 = self.conv7b(xup1)
        xup2 = self.conv5b(xup2)
        xup3 = self.conv3b(xup3)

        # up-path
        xup2 = self.upsample(xup3) + xup2
        xup1 = self.upsample(xup2) + xup1

        # Global Avg Pooling
        gp = self.GPool(x)
        gp = self.conv1p(gp)
        gp = F.upsample(gp, size=(H, W), mode='bilinear', align_corners=True)

        # Merge
        if mode == 'std':
            x1 = self.upsample(xup1) * x1
        elif mode == 'reduced':
            x1 = xup1 * x1
        elif mode == 'extended':
            x1 = F.upsample(xup1, scale_factor=4, mode='bilinear', align_corners=True) * x1
        x1 = x1 + gp
        return x1


class FeaturePyramidAttention_v2(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self.conv1 = ConvBn2d(channels, out_channels, kernel_size=1, stride=1, padding=0, act=True)
        self.conv1p = ConvBn2d(channels, out_channels, kernel_size=1, stride=1, padding=0, act=True)

        self.conv3a = ConvBn2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True)
        self.conv3b = ConvBn2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True)

        self.conv5a = ConvBn2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, act=True)
        self.conv5b = ConvBn2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, act=True)

        self.conv7a = ConvBn2d(channels, out_channels, kernel_size=7, stride=1, padding=3, act=True)
        self.conv7b = ConvBn2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, act=True)

        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, mode='std'):
        H, W = x.shape[2:]
        # down-path
        if mode == 'std':
            xup1 = self.downsample(x)
            xup1 = self.conv7a(xup1)
        elif mode == 'reduced':
            xup1 = self.conv7a(x)

        xup2 = self.downsample(xup1)
        xup2 = self.conv5a(xup2)

        xup3 = self.downsample(xup2)
        xup3 = self.conv3a(xup3)

        # Skips
        x1 = self.conv1(x)
        xup1 = self.conv7b(xup1)
        xup2 = self.conv5b(xup2)
        xup3 = self.conv3b(xup3)

        # up-path
        xup2 = self.upsample(xup3) + xup2
        xup1 = self.upsample(xup2) + xup1

        # Global Avg Pooling
        gp = self.GPool(x)
        gp = self.conv1p(gp)
        gp = F.upsample(gp, size=(H, W), mode='bilinear', align_corners=True)

        # Merge
        if mode == 'std':
            x1 = self.upsample(xup1) * x1
        elif mode == 'reduced':
            x1 = xup1 * x1
        x1 = x1 + gp
        return x1


class GlobalAttentionUpsample(nn.Module):
    def __init__(self, skip_channels, channels, out_channels=None):
        super(GlobalAttentionUpsample, self).__init__()
        self.out_channels = out_channels
        if out_channels is None:
            out_channels = channels
        self.conv3 = nn.Conv2d(skip_channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConvBn2d(channels, channels, kernel_size=1, padding=0)
        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if out_channels is not None:
            self.conv_out = ConvBn2d(channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip, up=True):
        # Reduce channels
        skip = self.conv3(skip)
        # Upsample
        if up:
            x = self.upsample(x)
        # GlobalPool and conv1
        cal1 = self.GPool(x)
        cal1 = self.conv1(cal1)
        cal1 = self.relu(cal1)

        # Calibrate skip connection
        skip = cal1 * skip
        # Add
        x = x + skip
        if self.out_channels is not None:
            x = self.conv_out(x)
        return x


class AttentionUpsample(nn.Module):
    def __init__(self, skip_channels, channels, n_classes, out_channels=None):
        super().__init__()
        self.out_channels = out_channels
        if out_channels is None:
            out_channels = channels
        self.conv3 = nn.Conv2d(skip_channels, channels, kernel_size=3, padding=1)
        self.rconv1 = nn.Conv2d(channels, n_classes, kernel_size=1, padding=0)
        self.gconv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if out_channels is not None:
            self.conv_out = ConvBn2d(channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip, up=True):
        # Reduce channels
        skip = self.conv3(skip)
        # Upsample
        if up:
            x = self.upsample(x)

        # GlobalPool and conv1
        gcalib = self.GPool(x)
        gcalib = self.gconv1(gcalib)
        gcalib = torch.sigmoid(gcalib)

        # RegionalPool
        rcalib = self.rconv1(x)
        rcalib = torch.sigmoid(rcalib)

        # Calibrate skip connection
        skip = (gcalib * skip) + (rcalib * skip)
        # Add
        x = x + skip
        if self.out_channels is not None:
            x = self.conv_out(x)
        return x, rcalib
