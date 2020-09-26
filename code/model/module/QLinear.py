
#import pdb
#import math
#import warnings
#import numpy as np
#import torch
# import pytorch modules
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function

# import customized quantize functions
from .quantize_functions import *

__all__ = ['QLinear']



class QLinear(nn.Linear):
    """ quantized linear
        parameters: same as nn.Linear but adding q_cfg as quantize config
    """

    def __init__(self, in_planes, out_planes, bias=True, q_cfg=None):
        super(QLinear, self).__init__(in_planes, out_planes, bias)

        self.quantize    = False
        self.bw_a        = q_cfg["bw"][0]        # acivation bit width
        self.bw_w        = q_cfg["bw"][1]        # weight bit width
        self.bw_g        = q_cfg["bw"][2]        # gradient bit width
        self.linear_a    = q_cfg["linear"][0]    
        self.linear_w    = q_cfg["linear"][1]    
        self.linear_g    = q_cfg["linear"][2]   
        self.signed      = q_cfg["signed"]       # bool of signed quantize
        self.stochastic  = q_cfg["stochastic"]   # bool of stochastic rounding 
        self.erange      = q_cfg["erange"]      
        self.group       = q_cfg["group"]
        self.level_a     = q_cfg["level"][0]
        self.level_w     = q_cfg["level"][1]
        self.level_g     = q_cfg["level"][2]
        self.hard        = q_cfg["hard"]

        self.quantize_a = Quantize_A
        self.quantize_w = Quantize_W
        self.quantize_e = Quantize_E

    def forward(self, input):

        #print("debug linear stochastic:",self.stochastic)
        if self.training:
            #print("debug linear training:",self.training)
            Qinput  = self.quantize_a.apply(input, self.bw_a, self.linear_a, False,
                      self.stochastic, self.erange[0], self.group[0], self.level_a, self.hard)
        else:
            #print("debug linear test set sto false")
            Qinput  = self.quantize_a.apply(input, self.bw_a, self.linear_a, False,
                      False, self.erange[0], self.group[0], self.level_a, self.hard)
        Qweight = self.quantize_w.apply(self.weight, self.bw_w, self.linear_w, 
                  self.signed, self.stochastic, self.erange[1], self.group[1], self.level_w, self.hard) 

        if self.quantize:
            output = F.linear(Qinput, Qweight, self.bias)
            if self.bw_g is not None:
                output = self.quantize_g.apply(output, self.bw_g, self.linear_g, 
                         self.signed, True, self.erange[2], self.group[2], 
                         self.level_g, self.hard)
        else:
            output = F.linear(input, self.weight, self.bias)

        return output


