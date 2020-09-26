#import pdb
#import numpy as np
#import ipdb
# import pytorch modules
#import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function

# import customized quantize functions
from .quantize_functions import *

__all__ = ['QConv2d']



class QConv2d(nn.Conv2d):
    """ quantized conv2d
        parameters: same as Conv2d but adding q_cfg as quantize config
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, q_cfg=None):
        super(QConv2d, self).__init__(in_planes, out_planes, kernel_size, stride, 
                padding, dilation, groups, bias)

        self.quantize     = False
        self.mbw          = q_cfg["bw"]            # mantissa bit width
        self.ebw          = q_cfg["level"]         # exponent bit width
        self.signed       = q_cfg["signed"]        # bool of signed quantize
        self.stochastic   = q_cfg["stochastic"]    # bool of stochastic rounding 
        self.group        = q_cfg["group"]
        self.g_scale_type = q_cfg["g_scale_type"]         
        self.value_type   = q_cfg["value_type"]

        self.quantize_a = Quantize_A
        self.quantize_w = Quantize_W
        self.quantize_e = Quantize_E

    def forward(self, input):         
        if self.quantize:             
            Qinput  = self.quantize_a.apply(input,                                                  
                                            self.mbw[0],                                                  
                                            self.ebw[0],                                                  
                                            self.signed,                                                 
                                            False,                                                  
                                            self.group[0],                                                  
                                            self.g_scale_type,                                                 
                                            self.value_type)
            Qweight = self.quantize_w.apply(self.weight, 
                                            self.mbw[1],                                                  
                                            self.ebw[1],                                                  
                                            self.signed, 
                                            False, 
                                            self.group[1], 
                                            self.g_scale_type, 
                                            self.value_type) 
            #print("debug qinput requires grad: ", Qinput.requires_grad)
            #print("debug qweight requires grad: ", Qweight.requires_grad)

            output = F.conv2d( Qinput, Qweight, self.bias, 
                               self.stride, self.padding, self.dilation, self.groups)
            if self.mbw[2] <30:
                output = self.quantize_e.apply( output, 
                                                self.mbw[2], 
                                                self.ebw[2], 
                                                self.signed, 
                                                self.stochastic, 
                                                self.group[2],
                                                self.g_scale_type, 
                                                self.value_type)
        else:
            output = F.conv2d(input, self.weight, self.bias, 
                              self.stride, self.padding, self.dilation, self.groups)

        return output



