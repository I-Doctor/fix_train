'''
Modified from https://github.com/pytorch/vision.git
'''
import math
from collections import OrderedDict

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vggnet'
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    #named_layers = OrderedDict()
    #count_layers = 1
    #count_conv = 0
    #for m in layers:
    #    if isinstance(m, nn.Conv2d):
    #        named_layers['conv' + str(count_layers) + '.' + str(count_conv)] = m
    #        count_conv += 1
    #    elif isinstance(m, nn.BatchNorm2d):
    #        named_layers['bn' + str(count_layers) + '.' + str(count_conv)] = m
    #    elif isinstance(m, nn.ReLU):
    #        named_layers['relu' + str(count_layers) + '.' + str(count_conv)] = m
    #    elif isinstance(m, nn.MaxPool2d):
    #        named_layers['maxpool' + str(count_layers)] = m
    #        count_layers += 1
    #        count_conv = 0
    return nn.Sequential(*layers)  #(named_layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

cfg_name = {
    '11': 'A',
    '13': 'B',
    '16': 'C',
    '19': 'D',
}


def vggnet(depth=11, num_classes=10, batchnorm=True, q_cfg=None):
    """VGG model """

    return VGG(make_layers(cfg[cfg_name[str(depth)]], batch_norm=batchnorm))


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "C")"""
    return VGG(make_layers(cfg['C']))


def vgg16_bn():
    """VGG 16-layer model (configuration "C") with batch normalization"""
    return VGG(make_layers(cfg['C'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'D') with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))
