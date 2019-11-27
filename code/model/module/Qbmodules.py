
import pdb
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function

__all__ = ['QConv2d', 'QLinear']


class Quantize_W(Function):
    ''' Function which quantizes w directly during both forward and backward
        Call it by Quantize_W.apply(weight, ...)
    '''

    @staticmethod
    def forward(ctx, input, num_bits=8, linear=None, signed=True, stochastic=True, 
                dequantize=True, group=False, level=2):
        ''' Forward function
                input           : unquantized weight       
                num_bits        : bw of quantized weight 
                linear          : None for linear
                signed          : signed quantize or not    
                stochastic      : add noise when quantize or not 
                dequantize      : dequantize 
                group           : inplace 
                level           : additional bits for unusual values
        '''

        output = input.clone()

        with torch.no_grad():
            max_value = input.max()  
            min_value = input.min() 

        # TODO: re-add true zero computation
        # using GEMMLOWP quantize, shift and scale real input to qmin~qmax
        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**(num_bits)
        half = 2.**(num_bits -level -1)
        hscale = 2.**(level)
        range_value = max_value - min_value
        zero_point = min_value
        scale = range_value / (qmax - qmin)
        #print("debug w quantize gemmlowp, scale", scale)
                         
        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            #print("debug w min max:", output.min(), output.max())
            mask = output.abs().gt(half-0.5)
            #ratio = mask.type(torch.float16).mean()
            #print ('weight',ratio)
            # further scale values larger than half to compress precision
            output = torch.where(mask, output.div(hscale), output)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            output.clamp_(qmin, qmax).round_() # quantize
            output = torch.where(mask, output.mul(hscale), output)
            if dequantize:
                output.mul_(scale).add_(zero_point - qmin * scale) # dequantize

        return output


    @staticmethod
    def backward(ctx, grad_output):
        r''' Backword function
                ctx         : ctx record info from forward
                grad_output : unquantized gradient of weight
        '''
        grad_input = grad_output

        return grad_input, None, None, None, None, None, None, None



class Quantize_A(Function):
    r''' This is the function which quantizes a directly during forward
        Call it by Quantize_A.apply(input, ...)
    '''

    '''
    def __init__(self, num_bits=8, linear=None, signed=True, 
                stochastic=False, dequantize=True, group=False, level=4):
        super(Quantize_A, self).__init__()
    '''

    @staticmethod
    def forward(ctx, input, num_bits=8, linear=None, signed=False, 
                stochastic=True, dequantize=True, group=False, level=1):

        ''' Function arguements:
                input       : unquantized input         
                num_bits    : bw of quantized input
                linear      : None for linear
                signed      : signed quantize or not    
                stochastic  : add noise when quantize or not 
                dequantize  : dequantize                
        '''

        output = input.clone()

        num_chunks = 16
        with torch.no_grad():
            if len(input.shape) == 4:
                B,C,H,W = input.shape
                a = input.transpose(0, 1).contiguous()  # C x B x H x W
                a = a.view(C, num_chunks, (B*H*W) // num_chunks)
                max_value = a.max(-1)[0].max()  # calculate max of maxs of C*num_chunks chunks
                min_value = a.min(-1)[0].min()  # calculate min of mins of C*num_chunks chunks
            else:
                B,C = input.shape
                max_value = input.max() 
                min_value = input.min()

        # TODO: re-add true zero computation
        # using GEMMLOWP quantize, shift and scale real input to qmin~qmax
        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**(num_bits)
        half = 2.**(num_bits -level -1) if signed else 2.**(num_bits -level)
        hscale = 2.**(level)
        range_value = max_value - min_value
        zero_point = min_value
        scale = range_value / (qmax - qmin)
        #print("debug a quantize gemmlowp, scale:", scale)
                    
        with torch.no_grad():
            (output.add_(qmin * scale - zero_point)).div_(scale)
            #print("debug a min max:", output.min(), output.max())
            #if np.isnan(output.min().cpu()):
            #    exit('ERROR activation data nan')
            mask = output.abs().gt(half-0.5)
            #ratio = mask.type(torch.float16).mean()
            #print ('activ',ratio)
            # further scale values larger than half to compress precision
            output = torch.where(mask, output.div(hscale), output)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            output.clamp_(qmin, qmax).round_() # quantize
            output = torch.where(mask, output.mul(hscale), output)
            if dequantize:
                output.mul_(scale).add_(zero_point - qmin * scale) # dequantize

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator

        grad_input = grad_output
        #torch.set_printoptions(precision=8)
        #print("debug grad of input:",grad_input.max(), grad_input.min(), grad_input.shape)

        return grad_input, None, None, None, None, None, None, None




class Quantize_G(Function):
    r''' This is the function which quantizes g directly during backward
        Call it by Quantize_G.apply(input, ...)
    '''

    @staticmethod
    def forward(ctx, input, num_bits=8, linear=None, signed=True, stochastic=True, 
                dequantize=True, group=False, level=4):
        ''' Function arguements:
                input       : activation input                     
                num_bits    : bw of quantized gradient 
                signed      : signed quantize or not    
                stochastic  : add noise when quantize or not 
                dequantize  : dequantize                
        '''
        # record info and return input when forward
        ctx.num_bits   = num_bits
        ctx.linear     = linear
        ctx.signed     = signed
        ctx.stochastic = stochastic
        ctx.dequantize = dequantize
        ctx.group      = False
        ctx.level      = level

        output = input.clone()

        return output


    @staticmethod
    def backward(ctx, grad_output):

        grad_input = grad_output.clone()

        num_chunks = 16
        if len(grad_output.shape) == 4:
            B,C,H,W = grad_output.shape
            g = grad_output.transpose(0, 1).contiguous()  # C x B x H x W
            g = g.view(C, num_chunks, (B*H*W) // num_chunks)
            max_value = g.max(-1)[0].max()  # calculate max of maxs of C*num_chunks chunks
            min_value = g.min(-1)[0].min()  # calculate min of mins of C*num_chunks chunks
        else:
            B,C = grad_output.shape
            max_value = grad_output.max() 
            min_value = grad_output.min()

        num_bits = ctx.num_bits + ctx.level # first scale for n+l bits
        qmin = -(2.**(num_bits - 1)) if ctx.signed else 0.
        qmax = qmin + 2.**(num_bits)
        half = 2.**(num_bits-ctx.level-1)   # half is near n bits number
        hscale = 2.**(ctx.level)
        # TODO: re-add true zero computation
        # using GEMMLOWP quantize, shift and scale real input to qmin~qmax
        range_value = max_value - min_value
        zero_point = min_value
        scale = range_value / (qmax - qmin)
        #print("debug g quantize gemmlowp, scale:", scale)

        with torch.no_grad():
            grad_input.add_(qmin * scale - zero_point).div_(scale)
            #print("debug g min max:", grad_input.min(), grad_input.max())
            mask = grad_input.abs().gt(half-0.5)
            #ratio = mask.type(torch.float16).mean()
            #print ('grad',ratio)
            # further scale values larger than half to compress precision
            grad_input = torch.where(mask, grad_input.div(hscale), grad_input)
            if ctx.stochastic:
                noise = grad_input.new(grad_input.shape).uniform_(-0.5, 0.5)
                grad_input.add_(noise)
            grad_input.clamp_(qmin, qmax).round_()  # quantize
            grad_input = torch.where(mask, grad_input.mul(hscale), grad_input)
            if ctx.dequantize:
                grad_input.mul_(scale).add_(zero_point - qmin * scale) # dequantize
            #print("debug grad of output, quantized:", grad_input.max(), grad_input.min())

        return grad_input, None, None, None, None, None, None, None



def wQuantize(g, bw=8, linear=None, signed=True, stochastic=True, dequantize=True, inplace=False, magnitude=None):
    ''' This is the function which does g quantize by calling Quantize_G
    '''
    return Quantize_W.apply(g, bw, signed, stochastic, dequantize, inplace, magnitude)


def aQuantize(g, bw=8, signed=True, stochastic=True, dequantize=True, inplace=False, magnitude=None):
    ''' This is the function which does g quantize by calling Quantize_G
    '''
    return Quantize_A.apply(g, bw, False, stochastic, dequantize, inplace, magnitude)


def gQuantize(g, bw=8, signed=True, stochastic=True, dequantize=True, inplace=False, magnitude=None):
    ''' This is the function which does g quantize by calling Quantize_G
    '''
    return Quantize_G.apply(g, bw, signed, stochastic, dequantize, inplace, magnitude)



class QConv2d(nn.Conv2d):
    """quantized conv2d
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, q_cfg=None):
        super(QConv2d, self).__init__(in_planes, out_planes, kernel_size, stride, padding,
                 dilation, groups, bias)

        self.quantize    = False
        self.bw_a        = q_cfg["bw"][0]           # acivation bit width
        self.bw_w        = q_cfg["bw"][1]           # weight bit width
        self.bw_g        = q_cfg["bw"][2]           # gradient bit width
        self.linear_a    = q_cfg["linear"][0]       
        self.linear_w    = q_cfg["linear"][1]       
        self.linear_g    = q_cfg["linear"][2]      
        self.signed      = q_cfg["signed"]          # bool of signed quantize
        self.stochastic  = q_cfg["stochastic"]      # bool of stochastic rounding during quantize
        self.dequantize  = q_cfg["dequantize"]      # bool of dequantize during the software simulation
        self.group       = q_cfg["group"]
        self.level_a     = q_cfg["level"][0]
        self.level_w     = q_cfg["level"][1]
        self.level_g     = q_cfg["level"][2]

        self.quantize_a = Quantize_A
        self.quantize_w = Quantize_W
        self.quantize_g = Quantize_G

    def forward(self, input):
        
        #print (input,self.bw_a, self.linear_a, self.signed,
        #          self.stochastic, self.dequantize, self.group, self.level_a)
        if self.train:
            Qinput  = self.quantize_a.apply(input, self.bw_a, self.linear_a, False,
                      self.stochastic, self.dequantize, self.group, self.level_a)
        else:
            Qinput  = self.quantize_a.apply(input, self.bw_a, self.linear_a, False,
                      False, self.dequantize, self.group, self.level_a)
        Qweight = self.quantize_w.apply(self.weight, self.bw_w, self.linear_w, 
                  self.signed, True, self.dequantize, self.group, self.level_w) 
        #print("debug qinput requires grad: ", Qinput.requires_grad)
        #print("debug qweight requires grad: ", Qweight.requires_grad)

        if self.quantize:
            output = F.conv2d(Qinput, Qweight, self.bias, 
                               self.stride, self.padding, self.dilation, self.groups)
            if self.bw_g is not None:
                output = self.quantize_g.apply(output, self.bw_g, self.linear_g, 
                         self.signed, True, self.dequantize, self.group,self.level_g)
        else:
            output = F.conv2d(input, self.weight, self.bias, 
                              self.stride, self.padding, self.dilation, self.groups)

        return output



class QLinear(nn.Linear):
    """quantized linear
    """

    def __init__(self, in_planes, out_planes, bias=True, q_cfg=None):
        super(QLinear, self).__init__(in_planes, out_planes, bias)

        self.quantize    = False
        self.bw_a        = q_cfg["bw"][0]           # acivation bit width
        self.bw_w        = q_cfg["bw"][1]           # weight bit width
        self.bw_g        = q_cfg["bw"][2]           # gradient bit width
        self.linear_a    = q_cfg["linear"][0]       
        self.linear_w    = q_cfg["linear"][1]       
        self.linear_g    = q_cfg["linear"][2]      
        self.signed      = q_cfg["signed"]          # bool of signed quantize
        self.stochastic  = q_cfg["stochastic"]      # bool of stochastic rounding during quantize
        self.dequantize  = q_cfg["dequantize"]      # bool of dequantize during the software simulation
        self.group       = q_cfg["group"]
        self.level_a     = q_cfg["level"][0]
        self.level_w     = q_cfg["level"][1]
        self.level_g     = q_cfg["level"][2]

        self.quantize_a = Quantize_A
        self.quantize_w = Quantize_W
        self.quantize_g = Quantize_G

    def forward(self, input):

        Qinput  = self.quantize_a.apply(input, self.bw_a, self.linear_a, False,
                  self.stochastic, self.dequantize, self.group, self.level_a)
        Qweight = self.quantize_w.apply(self.weight, self.bw_w, self.linear_w, 
                  self.signed, True, self.dequantize, self.group, self.level_w) 

        if self.quantize:
            output = F.linear(Qinput, Qweight, self.bias)
            if self.bw_g is not None:
                output = self.quantize_g.apply(output, self.bw_g, self.linear_g, 
                         self.signed, True, self.dequantize, self.group, 
                         self.level_g)
        else:
            output = F.linear(input, self.weight, self.bias)

        return output

