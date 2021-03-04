######################################################################
# (c) Copyright EFC of NICS, Tsinghua University. All rights reserved.
#  Author: Kai Zhong
#  Email : zhongk19@mails.tsinghua.edu.cn
#  
#  Create Date : 2020.11.16
#  File Name   : vggnet.py
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
    '''

    def __init__(self, depth, num_classes=10, batchnorm=True, q_cfg=None):

        assert depth in _depth
        self.depth = depth
        linear_channel = 512 if num_classes == 10 else 4096
        linear_input   = 1 if num_classes == 10 else 7
        linear_dropout = nn.Identity() #if num_classes == 10 else nn.Dropout()
        super(VGGNet, self).__init__()
         
        self.quantize = False
        self.features = self._make_features(_depth[self.depth], batchnorm, q_cfg)
        if q_cfg is not None and q_cfg.qlinear:
            self.classifier = nn.Sequential(
                QLinear(512 * linear_input * linear_input, linear_channel, 
                        q_cfg=q_cfg),
                nn.ReLU(True),
                linear_dropout,
                QLinear(linear_channel, linear_channel, q_cfg=q_cfg),
                nn.ReLU(True),
                linear_dropout,
                QLinear(linear_channel, num_classes, q_cfg=q_cfg),
            )
        else:
            print('totally no quantization linears')
            # TOTEST: dropout
            self.classifier = nn.Sequential(
                nn.Linear(512 * linear_input * linear_input, linear_channel),
                nn.ReLU(True),
                linear_dropout,
                nn.Linear(linear_channel, linear_channel),
                nn.ReLU(True),
                linear_dropout,
                nn.Linear(linear_channel, num_classes),
            )
            #self.classifier = nn.Linear(512 * linear_input * linear_input, num_classes)

        # initialize
        self._initialize()


    def _make_features(self, structure, batch_norm=True, q_cfg=None):
        ''' the function to make feature layers of vggnet
            
            structure   : class, block type of the layers set
            batch_norm  : batch norm or not
            q_cfg       : dict, how to do quantization
        '''
        layers = []
        in_channels = 3

        for i, v in enumerate(structure):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if q_cfg is not None and i!=0:
                    conv2d = QConv2d(in_channels, v, kernel_size=3, padding=1, q_cfg=q_cfg)
                    if batch_norm:
                        if q_cfg.qbn:
                            bn2d = QBatchNorm2d(v, q_cfg)
                        else:
                            bn2d = nn.BatchNorm2d(v)
                        layers += [conv2d, bn2d, nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                else:
                    print('totally no quantization conv')
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
                # TOTEST: initialize
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
        

    def enable_quantize(self):
        ''' API to enable quantize
        '''
        self.apply(q_enable)

    """
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
    """



def vggnet(depth, num_classes, batchnorm=True, q_cfg=None):
    ''' function to get a VGGNet
    '''
    return VGGNet(depth, num_classes, batchnorm, q_cfg)


