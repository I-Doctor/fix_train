#import torch
import torch.nn as nn
import math
from .module import *

__all__ = ['vgglikenet', 'VGGLNet']



class VGGLNet(nn.Module):
    ''' VGGLNet class define which support quantize
        
        batchnorm   : batch norm or not
        q_cfg       : dict, how to do quantization
    '''

    def __init__(self, batchnorm=True, q_cfg=None):

        super(SMPNet, self).__init__()
         
        self.quantize = False
        self.num_classes = 10

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False, q_cfg=q_cfg)
        self.bn1   = nn.BatchNorm2d(v)
        self.relu1 = nn.ReLU(inplace=True) 
        if q_cfg is not None:
            self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, q_cfg=q_cfg)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.relu2 = nn.ReLU(inplace=True) 

        self._make_features(batchnorm, q_cfg)

        if q_cfg is not None:
            self.classifier = nn.Sequential(
                QLinear(16 * 1 * 1, self.num_classes, q_cfg=q_cfg),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(16 * 1 * 1, self.num_classes),
            )

        # initialize
        self._initialize()


    def _make_features(self, batch_norm=True, q_cfg=None):
        ''' the function to make feature layers of vggnet
            
            batch_norm  : batch norm or not
            q_cfg       : dict, how to do quantization
        '''
        layers = []
        in_channels = 3
        v = 16

        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        for i in range(self.depth -1):

            if q_cfg is not None:
                conv2d = QConv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=False, q_cfg=q_cfg)
                bn     = nn.BatchNorm2d(v)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=False)
                bn     = nn.BatchNorm2d(v)
            if batch_norm:
                layers += [conv2d, bn, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.AvgPool2d(kernel_size=8)]

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

    def set_lr_scale(self, lr_p):
        ''' API to set learning rate scale bit
        '''
        for m in self.modules():
            if hasattr(m, 'lr_scale_p') and hasattr(m, "magnitude"):
                if m.magnitude == 'ceil' and lr_p != 0:
                    m.lr_scale_p = lr_p
                    print("    Set lr scale bit of")
                    print(m)



def simplenet(depth, num_classes, batchnorm=True, q_cfg=None):
    ''' function to get a SMPNet
    '''

    return SMPNet(depth, num_classes, batchnorm, q_cfg)

