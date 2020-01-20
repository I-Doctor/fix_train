
import pdb
import math
import warnings
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
                erange='max', group=False, level=1):
        ''' Forward function
                input           : unquantized weight       
                num_bits        : bw of quantized weight 
                linear          : None for linear
                signed          : signed quantize or not    
                stochastic      : add noise when quantize or not 
                group           : inplace 
                level           : additional bits for unusual values
        '''
        output = input.clone()

        if group != False:
            if len(input.shape) == 4:
                B,C,H,W = input.shape
                a = input.clone()
            else:
                B,C,H,W = (0,0,0,0)
                raise ValueError('appear fc')

            if group == 'nc':
                warnings.warn('You are using nc group for weight')
                a = a.view(B, C, H*W)
                if erange == 'max':
                    max_value = a.max(2)[0]
                    min_value = a.min(2)[0]
                elif erange == 'mean':
                    raise ValueError('wrong nc group work with mean erange')
                elif erange == 'std' :
                    raise ValueError('wrong nc group work with std erange')
                else:
                    raise ValueError('wrong erange config')
                assert len(max_value.shape) == 2
                max_value.unsqueeze_(2).unsqueeze_(2)
                min_value.unsqueeze_(2).unsqueeze_(2)
            else:
                #print('debug w a',a.shape)
                if group == 'n':
                    a = a.view(B, C, H*W)
                elif group == 'c':
                    a = a.transpose(0,1).contiguous().view(C, B, H*W)
                elif group == 'nt':
                    a = a.view(B//2, C*2, H*W)
                else:
                    raise ValueError('wrong group config')

                #print('debug w a.view',a.shape)
                if erange == 'max':
                    max_value = a.max(-1)[0].max(-1)[0]
                    min_value = a.min(-1)[0].min(-1)[0]
                elif erange == 'mean':
                    max_value = a.max(-1)[0].mean(-1)
                    min_value = a.min(-1)[0].mean(-1)
                elif erange == 'std':
                    mean_value = a.mean((1,2))
                    std_value = a.std((1,2))
                    max_value = mean_value + 3*std_value
                    min_value = mean_value - 3*std_value
                else:
                    raise ValueError('wrong erange config')
                #print('debug w',max_value.shape)
                assert len(max_value.shape) == 1
                if group == 'c':
                    max_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(0)
                    min_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(0)
                elif group == 'nt':
                    max_value = max_value.repeat(1,2).view(B,1,1,1)
                    min_value = min_value.repeat(1,2).view(B,1,1,1)
                else:
                    max_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(1)
                    min_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(1)

        elif group == False:       # group == False
            if erange == 'max':
                max_value = input.max()  
                min_value = input.min() 
            elif erange == 'mean':
                max_value = input.max(-1)[0].max(-1)[0].max(-1)[0].mean()
                min_value = input.min(-1)[0].min(-1)[0].min(-1)[0].mean()
            elif erange == 'std':
                mean_value = input.mean()
                std_value = input.std()
                max_value = mean_value + 3*std_value
                min_value = mean_value - 3*std_value
            else:
                raise ValueError('wrong erange config')
        else:
            raise ValueError('wrong group config')


        # TODO: re-add true zero computation
        # using GEMMLOWP quantize, shift and scale real input to qmin~qmax
        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**(num_bits)
        qrange = qmax-qmin
        #range_value = max_value - min_value
        abs_max = torch.max(max_value, -min_value)
        range_value = 2*abs_max
        zero_point = -1*abs_max
        scale = range_value / qrange
        half = 2.**(num_bits -level -1)
        hscale = 2.**(level)
                         
        if linear is None:
            output.add_(qmin * scale - zero_point).div_(scale)
            #print("debug w min max:", output.min(), output.max())
            mask = output.abs().gt(half-0.5)
            #ratio = mask.type(torch.float16).mean()
            #print ('weight',ratio)
            # further scale values larger than half to compress precision
            output = torch.where(mask, output.div(hscale), output)
            if stochastic:
                #print("debug stochastic")
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            output.clamp_(qmin, qmax).round_() # quantize
            output = torch.where(mask, output.mul(hscale), output)
            output.mul_(scale).add_(zero_point - qmin * scale) # dequantize
        elif linear == 'tanh':
            output.add_(qmin * scale - zero_point).div_(scale)
            mask = output.abs().gt(half-0.5)
            #ratio = mask.type(torch.float16).mean()
            #print ('weight',ratio)
            # further scale values larger than half to compress precision
            output = torch.where(mask, output.div(hscale), output)

            ratio = 4/half
            thres = half
            output.mul_(ratio).sigmoid_().mul_(2*thres).sub_(thres)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            output.clamp_(qmin,qmax).round_()
            output.add_(thres).div_(2*thres).reciprocal_().sub_(1).log_().div(-ratio)

            output = torch.where(mask, output.mul(hscale), output)
            output.mul_(scale).add_(zero_point - qmin * scale) # dequantize

        else:
            raise ValueError('wrong linear config')

        return output

    @staticmethod
    def backward(ctx, grad_output):
        r''' Backword function
                ctx         : ctx record info from forward
                grad_output : unquantized gradient of weight
        '''
        grad_input = grad_output.clone()

        return grad_input, None, None, None, None, None, None, None



class Quantize_A(Function):
    r''' This is the function which quantizes a directly during forward
        Call it by Quantize_A.apply(input, ...)
    '''

    @staticmethod
    def forward(ctx, input, num_bits=8, linear=None, signed=False, 
                stochastic=True, erange='max', group=False, level=2):
        ''' Function arguements:
                input       : unquantized input         
                num_bits    : bw of quantized input
                linear      : None for linear
                signed      : signed quantize or not    
                stochastic  : add noise when quantize or not 
        '''
        output = input.clone()

        num_chunks = 16

        if group != False:
            if len(input.shape) == 4:
                B,C,H,W = input.shape
                a = input.clone()
            else:
                B,C,H,W = (0,0,0,0)
                raise ValueError('appear fc')

            if group == 'nc':
                #print("debug nc ",a.shape)
                a = a.view(B, C, H*W)
                #print("debug nc ",a.shape)
                if erange == 'max':
                    max_value = a.max(-1)[0]
                    min_value = a.min(-1)[0]
                elif erange == 'mean':
                    raise ValueError('wrong nc group work with mean erange')
                elif erange == 'std':
                    raise ValueError('wrong nc group work with std erange')
                else:
                    raise ValueError('wrong erange config')
                #print("debug nc ",max_value.shape)
                assert len(max_value.shape) == 2
                max_value.unsqueeze_(2).unsqueeze_(2)
                min_value.unsqueeze_(2).unsqueeze_(2)
                #print("debug nc ",max_value.shape)
            elif B > 1:
                if ((B*H*W) % num_chunks != 0) or ((C*H*W) % num_chunks != 0):
                    num_chunks =1
                if group == 'n':
                    a = a.view(B, num_chunks, (C*H*W)//num_chunks)
                elif group == 'c':
                    a = a.transpose(0,1).contiguous().view(C,num_chunks,(B*H*W)//num_chunks)
                elif group == 'nt':
                    a = a.view(B//2, num_chunks, (C*2*H*W)//num_chunks)
                else:
                    raise ValueError('wrong group config')

                if erange == 'max':
                    max_value = a.max(-1)[0].max(-1)[0]
                    min_value = a.min(-1)[0].min(-1)[0]
                elif erange == 'mean':
                    max_value = a.max(-1)[0].mean(-1)
                    min_value = a.min(-1)[0].mean(-1)
                elif erange == 'std':
                    mean_value = a.mean((1,2))
                    std_value = a.std((1,2))
                    max_value = mean_value + 3*std_value
                    min_value = mean_value - 3*std_value
                else:
                    raise ValueError('wrong erange config')
                assert len(max_value.shape) == 1
                if group == 'c':
                    max_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(0)
                    min_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(0)
                elif group == 'nt':
                    max_value = max_value.repeat(1,2).view(B,1,1,1)
                    min_value = min_value.repeat(1,2).view(B,1,1,1)
                else:
                    max_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(1)
                    min_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(1)
            else:  # B == 1
                max_value = input.max()  
                min_value = input.min() 
                print('b==1',end="")

        elif group == False:       # group == False
            if erange == 'max':
                max_value = input.max()  
                min_value = input.min() 
            elif erange == 'mean':
                max_value = input.max(-1)[0].max(-1)[0].max(-1)[0].mean()
                min_value = input.min(-1)[0].min(-1)[0].min(-1)[0].mean()
            elif erange == 'std':
                mean_value = input.mean()
                std_value = input.std()
                max_value = mean_value + 3*std_value
                min_value = mean_value - 3*std_value
            else:
                raise ValueError('wrong erange config')
        else:
            raise ValueError('wrong group config')

        # using GEMMLOWP quantize, shift and scale real input to qmin~qmax
        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**(num_bits)
        qrange = qmax - qmin
        #range_value = max_value - min_value
        #zero_point = min_value
        abs_max = torch.max(max_value, -min_value)
        range_value = 2*abs_max
        zero_point = -1*abs_max
        scale = range_value / qrange
        half = 2.**(num_bits -level -1) if signed else 2.**(num_bits -level)
        hscale = 2.**(level)
                    
        if linear is None:
            (output.add_(qmin * scale - zero_point)).div_(scale)
            #print("debug a min max:", output.min(), output.max())
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
            output.mul_(scale).add_(zero_point - qmin * scale) # dequantize
        elif linear == 'tanh':
            output.add_(qmin * scale - zero_point).div_(scale)
            mask = output.abs().gt(half-0.5)
            #ratio = mask.type(torch.float16).mean()
            #print ('weight',ratio)
            # further scale values larger than half to compress precision
            output = torch.where(mask, output.div(hscale), output)

            ratio = 4/half
            thres = half
            output.mul_(ratio).sigmoid_().mul_(2*thres).sub_(thres)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            output.clamp_(qmin,qmax).round_()
            output.add_(thres).div_(2*thres).reciprocal_().sub_(1).log_().div(-ratio)

            output = torch.where(mask, output.mul(hscale), output)
            output.mul_(scale).add_(zero_point - qmin * scale) # dequantize
        else:
            raise ValueError('wrong linear config')

        return output


    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator

        grad_input = grad_output.clone()
        #torch.set_printoptions(precision=8)
        #print("debug grad of input:",grad_input.max(), grad_input.min(), grad_input.shape)

        return grad_input, None, None, None, None, None, None, None




class Quantize_G(Function):
    r''' This is the function which quantizes g directly during backward
        Call it by Quantize_G.apply(input, ...)
    '''

    @staticmethod
    def forward(ctx, input, num_bits=8, linear=None, signed=True, stochastic=True, 
                erange=True, group=False, level=3):
        ''' Function arguements:
                input       : activation input                     
                num_bits    : bw of quantized gradient 
                signed      : signed quantize or not    
                stochastic  : add noise when quantize or not 
        '''
        # record info and return input when forward
        ctx.num_bits   = num_bits
        ctx.linear     = linear
        ctx.signed     = signed
        ctx.stochastic = stochastic
        ctx.erange     = erange
        ctx.group      = group
        ctx.level      = level

        output = input.clone()

        return output


    @staticmethod
    def backward(ctx, input):

        output = input.clone()

        num_chunks = 16

        if ctx.group != False:
            if len(input.shape) == 4:
                B,C,H,W = input.shape
                a = input.clone()
            else:
                B,C,H,W = (0,0,0,0)
                raise ValueError('appear fc')

            if ctx.group == 'nc':
                a = a.view(B, C, H*W)
                if ctx.erange == 'max':
                    max_value = a.max(2)[0]
                    min_value = a.min(2)[0]
                elif ctx.erange == 'mean':
                    raise ValueError('wrong nc group work with mean erange')
                elif ctx.erange == 'std':
                    raise ValueError('wrong nc group work with std erange')
                else:
                    raise ValueError('wrong erange config')
                assert len(max_value.shape) == 2
                max_value.unsqueeze_(2).unsqueeze_(2)
                min_value.unsqueeze_(2).unsqueeze_(2)
            elif B>1:
                if ((B*H*W) % num_chunks != 0) or ((C*H*W) % num_chunks != 0):
                    num_chunks =1
                if ctx.group == 'n':
                    a = a.view(B, num_chunks, (C*H*W)//num_chunks)
                elif ctx.group == 'c':
                    a = a.transpose(0,1).contiguous().view(C,num_chunks,(B*H*W)//num_chunks)
                elif ctx.group == 'nt':
                    a = a.view(B//2, num_chunks, (C*2*H*W)//num_chunks)
                else:
                    raise ValueError('wrong group config')

                if ctx.erange == 'max':
                    max_value = a.max(-1)[0].max(-1)[0]
                    min_value = a.min(-1)[0].min(-1)[0]
                elif ctx.erange == 'mean':
                    max_value = a.max(-1)[0].mean(-1)
                    min_value = a.min(-1)[0].mean(-1)
                elif ctx.erange == 'std':
                    mean_value = a.mean((1,2))
                    std_value = a.std((1,2))
                    max_value = mean_value + 3*std_value
                    min_value = mean_value - 3*std_value
                else:
                    raise ValueError('wrong erange config')
                assert len(max_value.shape) == 1
                if ctx.group == 'c':
                    max_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(0)
                    min_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(0)
                elif ctx.group == 'nt':
                    max_value = max_value.repeat(1,2).view(B,1,1,1)
                    min_value = min_value.repeat(1,2).view(B,1,1,1)
                else:
                    max_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(1)
                    min_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(1)
            else:  # B == 1
                max_value = input.max()  
                min_value = input.min() 
                print('b==1',end="")
                    

        elif ctx.group == False:       # group == False
            if ctx.erange == 'max':
                max_value = input.max()  
                min_value = input.min() 
            elif ctx.erange == 'mean':
                max_value = input.max(-1)[0].max(-1)[0].max(-1)[0].mean()
                min_value = input.min(-1)[0].min(-1)[0].min(-1)[0].mean()
            elif ctx.erange == 'std':
                mean_value = input.mean()
                std_value = input.std()
                max_value = mean_value + 3*std_value
                min_value = mean_value - 3*std_value
            else:
                raise ValueError('wrong erange config')
        else:
            raise ValueError('wrong group config')

        num_bits = ctx.num_bits + ctx.level # first scale for n+l bits
        qmin = -(2.**(num_bits - 1)) if ctx.signed else 0.
        qmax = qmin + 2.**(num_bits)
        qrange = qmax - qmin
        #zero_point = min_value
        #range_value = max_value - min_value
        abs_max = torch.max(max_value, -min_value)
        range_value = 2*abs_max
        zero_point = -1*abs_max
        scale = (range_value/qrange)
        half = 2.**(num_bits-ctx.level-1)   # half is near n bits number
        hscale = 2.**(ctx.level)

        if ctx.linear is None:
            output.add_(qmin * scale - zero_point).div_(scale)
            mask = output.abs().gt(half-0.5)
            #ratio = mask.type(torch.float16).mean()
            #print ('grad',ratio)
            # further scale values larger than half to compress precision
            output = torch.where(mask, output.div(hscale), output)
            if ctx.stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            output.clamp_(qmin, qmax).round_()  # quantize
            output = torch.where(mask, output.mul(hscale), output)
            output.mul_(scale).add_(zero_point - qmin * scale) # dequantize
        elif ctx.linear == 'tanh':
            output.add_(qmin * scale - zero_point).div_(scale)
            mask = output.abs().gt(half-0.5)
            #ratio = mask.type(torch.float16).mean()
            #print ('weight',ratio)
            # further scale values larger than half to compress precision
            output = torch.where(mask, output.div(hscale), output)

            ratio = 4/half
            thres = half
            output.mul_(ratio).sigmoid_().mul_(2*thres).sub_(thres)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            output.clamp_(qmin,qmax).round_()
            output.add_(thres).div_(2*thres).reciprocal_().sub_(1).log_().div(-ratio)

            output = torch.where(mask, output.mul(hscale), output)
            output.mul_(scale).add_(zero_point - qmin * scale) # dequantize
        elif ctx.linear == 'lap':
            output.add_(qmin * scale - zero_point).div_(scale)
            mask = output.abs().gt(half-0.5)
            #ratio = mask.type(torch.float16).mean()
            #print ('weight',ratio)
            # further scale values larger than half to compress precision
            output = torch.where(mask, output.div(hscale), output)

            ratio = 4/half
            thres = half
            output.mul_(ratio).sigmoid_().mul_(2*thres).sub_(thres)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            output.clamp_(qmin,qmax).round_()
            output.add_(thres).div_(2*thres).reciprocal_().sub_(1).log_().div(-ratio)

            output = torch.where(mask, output.mul(hscale), output)
            output.mul_(scale).add_(zero_point - qmin * scale) # dequantize
        else:
            raise ValueError('wrong linear config')

        return output, None, None, None, None, None, None, None



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
        super(QConv2d, self).__init__(in_planes, out_planes, kernel_size, stride, 
                padding, dilation, groups, bias)

        self.quantize    = False
        self.bw_a        = q_cfg["bw"][0]         # acivation bit width
        self.bw_w        = q_cfg["bw"][1]         # weight bit width
        self.bw_g        = q_cfg["bw"][2]         # gradient bit width
        self.linear_a    = q_cfg["linear"][0]     
        self.linear_w    = q_cfg["linear"][1]     
        self.linear_g    = q_cfg["linear"][2]    
        self.signed      = q_cfg["signed"]        # bool of signed quantize
        self.stochastic  = q_cfg["stochastic"]    # bool of stochastic rounding 
        self.erange      = q_cfg["erange"]        
        self.group       = q_cfg["group"]
        self.level_a     = q_cfg["level"][0]
        self.level_w     = q_cfg["level"][1]
        self.level_g     = q_cfg["level"][2]

        self.quantize_a = Quantize_A
        self.quantize_w = Quantize_W
        self.quantize_g = Quantize_G

    def forward(self, input):
        
        #print("debug conv stochastic:",self.stochastic)
        if self.training:
            #print("debug self.train:",self.training)
            Qinput  = self.quantize_a.apply(input, self.bw_a, self.linear_a, False,
                      self.stochastic, self.erange[0], self.group[0], self.level_a)
        else:
            #print("debug conv test sto false")
            Qinput  = self.quantize_a.apply(input, self.bw_a, self.linear_a, False,
                      False, self.erange[0], self.group[0], self.level_a)
        Qweight = self.quantize_w.apply(self.weight, self.bw_w, self.linear_w, 
                  self.signed, self.stochastic, self.erange[1], self.group[1], self.level_w) 
        #print("debug qinput requires grad: ", Qinput.requires_grad)
        #print("debug qweight requires grad: ", Qweight.requires_grad)

        if self.quantize:
            output = F.conv2d(Qinput, Qweight, self.bias, 
                               self.stride, self.padding, self.dilation, self.groups)
            if self.bw_g is not None:
                output = self.quantize_g.apply(output, self.bw_g, self.linear_g, 
                         self.signed, True, self.erange[2], self.group[2],self.level_g)
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

        self.quantize_a = Quantize_A
        self.quantize_w = Quantize_W
        self.quantize_g = Quantize_G

    def forward(self, input):

        print("debug linear stochastic:",self.stochastic)
        if self.training:
            print("debug linear training:",self.training)
            Qinput  = self.quantize_a.apply(input, self.bw_a, self.linear_a, False,
                      self.stochastic, self.erange[0], self.group[0], self.level_a)
        else:
            print("debug linear test set sto false")
            Qinput  = self.quantize_a.apply(input, self.bw_a, self.linear_a, False,
                      False, self.erange[0], self.group[0], self.level_a)
        Qweight = self.quantize_w.apply(self.weight, self.bw_w, self.linear_w, 
                  self.signed, self.stochastic, self.erange[1], self.group[1], self.level_w) 

        if self.quantize:
            output = F.linear(Qinput, Qweight, self.bias)
            if self.bw_g is not None:
                output = self.quantize_g.apply(output, self.bw_g, self.linear_g, 
                         self.signed, True, self.erange[2], self.group[2], 
                         self.level_g)
        else:
            output = F.linear(input, self.weight, self.bias)

        return output

