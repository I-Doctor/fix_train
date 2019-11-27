######################################################################
# (c) Copyright EFC of NICS, Tsinghua University. All rights reserved.
#  Author: Kai Zhong
#  Email : zhongk19@mails.tsinghua.edu.cn
#  
#  Create Date : 2019.01.16
#  File Name   : vgg.py
#  Description : 
#  Dependencies: 
#  reference   https://github.com/
#  and         https://github.com/
######################################################################

import torch
import torch.nn as nn
import math
from .module import *

__all__ = ['vggnet', 'VGGNet']


_depth = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGNet(nn.Module):
    ''' VGGNet class define which support quantize and asparse
        
        depth       : depth of the network
        num_classes : how many classed does the network used to classifying
        batchnorm   : batch norm or not
        q_cfg       : dict, how to do quantization
        s_cfg       : dict, how to do asparse
    '''

    def __init__(self, depth, num_classes=10, batchnorm=True,
                 q_cfg=None, s_cfg=None):

        assert depth in _depth
        self.depth = depth
        super(VGGNet, self).__init__()
         
        self.quantize = False
        self.features = self._make_features(_depth[self.depth],batchnorm,q_cfg,s_cfg)
        if q_cfg is not None:
            self.classifier = nn.Sequential(
                QLinear(512 * 1 * 1, 512, q_cfg=q_cfg),
                nn.ReLU(True),
                nn.Dropout(),
                QLinear(512, 512, q_cfg=q_cfg),
                nn.ReLU(True),
                nn.Dropout(),
                QLinear(512, num_classes, q_cfg=q_cfg),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 1 * 1, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, num_classes),
            )

        # initialize
        self._initialize()


    def _make_features(self, structure, batch_norm=True, q_cfg=None, s_cfg=None):
        ''' the function to make feature layers of vggnet
            
            structure   : class, block type of the layers set
            batch_norm  : batch norm or not
            q_cfg       : dict, how to do quantization
            s_cfg       : dict, how to do asparse
        '''
        layers = []
        in_channels = 3

        for v in structure:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if q_cfg is not None:
                    conv2d = QConv2d(in_channels, v, kernel_size=3, padding=1, q_cfg=q_cfg)
                    if batch_norm:
                        layers += [conv2d, QBatchNorm2d(v, q_cfg=q_cfg), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)


    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(8. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
        

    def enable_quantize(self):
        ''' API to enable quantize
        '''
        self.apply(q_enable)

    def enable_asparse(self):
        ''' API to enable asparse
        '''
        self.apply(s_enable)

    def output_asparse(self):
        ''' API to output asparsity
        '''
        self.apply(sparse_output)

    def set_lr_scale(self, lr_p):
        ''' API to set learning rate scale bit
        '''
        for m in self.modules():
            if hasattr(m, 'lr_scale_p') and hasattr(m, "magnitude"):
                if m.magnitude == 'ceil' and lr_p != 0:
                    m.lr_scale_p = lr_p
                    print("    Set lr scale bit of")
                    print(m)



def vggnet(depth, num_classes, batchnorm=True, q_cfg=None, s_cfg=None, p_cfg=None):
    ''' function to get a VGGNet
    '''

    if p_cfg is None:
        return VGGNet(depth, num_classes, batchnorm, q_cfg, s_cfg)
    else:
        Pvggnet = VGGNet(depth, num_classes, batchnorm, q_cfg, s_cfg)
        return Pvnnnet


