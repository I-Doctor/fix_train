
#import pdb
#import math
#import warnings
#import numpy as np
import torch
#import torch.nn as nn
#import torch.nn.functional as F
from torch.autograd.function import Function

__all__ = ['Quantize_W', 'Quantize_A', 'Quantize_E']



def min_max(x, group):
    ''' calculate min max or mins maxs for group-wise with input tensor
    '''
    if group != False:          
        if len(x.shape) == 4:
            B,C,H,W = x.shape
        else:
            B,C,X = x.shape
            #raise ValueError('appear fc')
        if group == 'nc':
            x = x.view(B, C, -1)
            max_value = x.max(2)[0]
            min_value = x.min(2)[0]
            assert len(max_value.shape) == 2
            max_value.unsqueeze_(2).unsqueeze_(2)
            min_value.unsqueeze_(2).unsqueeze_(2)
        else:
            if group == 'n':
                x = x.view(B, C, -1)
            elif group == 'c':
                x = x.transpose(0,1).contiguous().view(C, B, -1)    
            else:
                raise ValueError('wrong group config')            
            max_value = x.max(-1)[0].max(-1)[0]
            min_value = x.min(-1)[0].min(-1)[0]
            assert len(max_value.shape) == 1        
            if group == 'c':
                max_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(0)
                min_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(0)
            else:
                max_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(1)
                min_value.unsqueeze_(1).unsqueeze_(1).unsqueeze_(1)
    else:
        max_value = x.max()  
        min_value = x.min() 
    max_value.clamp_(1e-10,1e10)        #non-zero
    
    return max_value, min_value



def simulate_quantize(a, max_value, min_value, mbits, ebits, signed, stochastic, group, g_scale_type, value_type):
    ''' simulate the quantization with computation
    '''
    # get scale_t
    abs_max   = torch.max(max_value, -min_value)         
    scale_t   = abs_max.max()
    scale_g   = 1.

    # get scale_g
    if group != False:
        real_scale_g = abs_max/scale_t
        exp_g = real_scale_g.log2().ceil_()
        simple_scale_g = 2.**exp_g
        scale_g = simple_scale_g
        if g_scale_type == 'complex':
            man_g = real_scale_g.div(simple_scale_g)
            man_g.mul_(4).ceil_().div_(4)
            complex_scale_g = man_g.mul(simple_scale_g)
            scale_g = complex_scale_g
    else:
        scale_g = 1.
    
    # divided by scales
    a.div_(scale_t).div_(scale_g)
    
    # quantize man_e and exp_e
    qmax = 2.**(mbits-1) if signed else 2.**(mbits)
    sign_e = torch.sign(a)         
    a.abs_().clamp_(1e-12,100)     # clamp zero values 
    exp_e  = a.log2().floor_()     # get exponent
    man_e  = 0
    mask   = 0
    # sudden underflow, normal value with min exp_e
    if value_type == 'sudden':
        exp_e.clamp_(-(2.**ebits),10)  # quantize exp_e by clamp small value
        man_e  = a.div(2.**exp_e)
        man_e.sub_(1).mul_(qmax)
    # progress underflow, unormal value with min exp_e + 1
    elif value_type == 'progress':
        exp_e.clamp_(-(2.**ebits)+1,10)  # quantize exp_e by clamp small value
        man_e  = a.div(2.**exp_e)
        # quantize mantissa of values greater than 1 by sub 1
        # quantize significant of values less than 1 and no sub 1
        mask   = man_e.gt(1)
        man_e  = torch.where(mask, man_e.sub(1), man_e)
        man_e.mul_(qmax)
    else:
        raise ValueError('wrong value type config')
    # quantize qmax scaled mantissa
    if stochastic:
        noise = man_e.new(man_e.shape).uniform_(-0.5, 0.5)
        man_e.add_(noise)
    man_e.round_()
    zero_e = man_e.gt(-0.5).float()     # underflow
    man_e.clamp_(0, qmax)
    # rescale qmax and add 1 back
    if value_type == 'sudden':
        man_e.div_(qmax).add_(1)
    elif value_type == 'progress':
        man_e.div_(qmax)
        man_e = torch.where(mask, man_e.add(1), man_e)
    else:  
        raise ValueError('wrong value type config')
    # dequantize to real float number
    a = zero_e.mul(sign_e.mul_(man_e).mul_(2.**(exp_e)))
    # rescale by scales
    a.mul_(scale_g).mul_(scale_t)
    
    return a, scale_t, scale_g



class Quantize_W(Function):
    ''' Function which quantizes w directly during both forward and backward
        Call it by Quantize_W.apply(weight, ...)
    '''
    @staticmethod
    def forward(ctx, input, mbits=4, ebits=2, signed=False, stochastic=False, 
                group='n', g_scale_type='simple', value_type='sudden'):
        ''' Forward function
                input           : unquantized weight       
                mbits           : mantissa bw of quantized weight
                ebits           : exponent bw of quantized weight
                signed          : signed quantize means no extra sign bit    
                stochastic      : add noise when quantize or not 
                group           : how to group 
        '''
        a = input.clone()
        aa = a.clone()
        max_value, min_value = min_max(input, group)
        qa, scale_t, scale_g = simulate_quantize(a, max_value, min_value, 
                                                 mbits, 
                                                 ebits, 
                                                 signed, 
                                                 stochastic, 
                                                 group, 
                                                 g_scale_type,
                                                 value_type)
        
        return qa      
        

    @staticmethod
    def backward(ctx, grad_output):
        r''' Backword function
                ctx         : ctx record info from forward
                grad_output : unquantized gradient of weight
        '''
        grad_input = grad_output.clone()

        return grad_input, None, None, None, None, None, None, None, None



class Quantize_A(Function):
    ''' Function which quantizes a directly during both forward and backward
        Call it by Quantize_A.apply(input, ...)
    '''
    @staticmethod
    def forward(ctx, input, mbits=4, ebits=2, signed=False, stochastic=False, 
                group='nc', g_scale_type='simple', value_type='sudden'):
        ''' Forward function
                input           : unquantized weight       
                mbits           : mantissa bw of quantized weight
                ebits           : exponent bw of quantized weight
                signed          : signed quantize means no extra sign bit    
                stochastic      : add noise when quantize or not 
                group           : how to group 
        '''
        a = input.clone()
        aa = a.clone()
        max_value, min_value = min_max(input, group)
        qa, scale_t, scale_g = simulate_quantize(a, max_value, min_value, 
                                                 mbits, 
                                                 ebits, 
                                                 signed, 
                                                 stochastic, 
                                                 group, 
                                                 g_scale_type,
                                                 value_type)
        ctx.scale_t = scale_t
        
        return qa 
        

    @staticmethod
    def backward(ctx, grad_output):
        r''' Backword function
                ctx         : ctx record info from forward
                grad_output : unquantized gradient of weight
        '''
        grad_input = grad_output.clamp(-ctx.scale_t, ctx.scale_t)

        return grad_input, None, None, None, None, None, None, None, None



class Quantize_E(Function):
    ''' Function which quantizes e directly during backward
        Call it by Quantize_E.apply(input, ...) in forward process
    '''
    @staticmethod
    def forward(ctx, input, mbits=4, ebits=2, signed=False, stochastic=True, 
                group='nc', g_scale_type='simple', value_type='sudden'):
        ''' Forward function
                input           : unquantized weight       
                mbits           : mantissa bw of quantized weight
                ebits           : exponent bw of quantized weight
                signed          : signed quantize means no extra sign bit    
                stochastic      : add noise when quantize or not 
                group           : how to group 
        '''
        # record info and return input when forward
        ctx.mbits        = mbits
        ctx.ebits        = ebits
        ctx.signed       = signed
        ctx.stochastic   = stochastic
        ctx.group        = group
        ctx.g_scale_type = g_scale_type
        ctx.value_type   = value_type

        output = input.clone()

        return output


    @staticmethod
    def backward(ctx, grad_output):
        ''' Backword function
            ctx         : ctx record info from forward                 
            grad_output : unquantized gradient of activation         
        '''
        a = grad_output.clone()         
        aa = a.clone()         
        max_value, min_value = min_max(grad_output, ctx.group)       
        qa, scale_t, scale_g = simulate_quantize(a, max_value, min_value,  
                                                 ctx.mbits,     
                                                 ctx.ebits, 
                                                 ctx.signed,
                                                 ctx.stochastic,
                                                 ctx.group, 
                                                 ctx.g_scale_type,
                                                 ctx.value_type)
        
        return qa, None, None, None, None, None, None, None, None



if __name__ == '__main__':

    x = torch.tensor([[[[1.00,2.00,3.00,4.00],
                        [5.00,6.00,7.00,8.00]],
                       [[9.00,10.0,11.0,12.0],
                        [13.0,14.0,15.0,16.0]]],
                      [[[17.0,18.0,19.0,20.0],
                        [21.0,22.0,23.0,24.0]],
                       [[25.0,26.0,27.0,28.0],
                        [29.0,30.0,31.0,32.0]]]])
    xx = x
    x.requires_grad = True
    print(x)
    print(x.shape)
#    print(xx)
#    print(xx.shape)
#    y = Quantize_A.apply(x, 3, None, False, False, 'max', 'nc', 1)
#    y = Quantize_W.apply(x, 3, None, False, False, 'max', 'nc', 1)
    y = Quantize_G.apply(x, 2, None, False, False, 'max', 'nc', 1)
    z = (y*xx).sum()
    xf=z.backward()
    print(xf)

