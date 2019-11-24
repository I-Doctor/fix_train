# coding:utf-8
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from .summary import get_names_dict

__all__ = ['hook_insert']


def hook_insert(model, type='a', layer_names=None, out_data_list=None):
    """
    Inset hooks into pytorch model

    """

    def register_hook(module):

        def a_hook(module, input, output):
            name = ''
            for key, val in names.items():
                #print(val)
                #print(module)
                if val == module:
                    name = key
            # <class 'torch.nn.modules.conv.Conv2d'>
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(out_data_list)
            m_key = module_idx + 1
            out_data_list[m_key] = OrderedDict()
            out_data_list[m_key]['name'] = name
            out_data_list[m_key]['class_name'] = class_name
            out_data_list[m_key]['output_a'] = output.data.cpu().numpy() 
            
        def g_hook(module, input_grad, output_grad):
            name = ''
            for key, val in names.items():
                if val == module:
                    name = key
            # <class 'torch.nn.modules.conv.Conv2d'>
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(out_data_list)
            m_key = module_idx + 1
            out_data_list[m_key] = OrderedDict()
            out_data_list[m_key]['name'] = name
            out_data_list[m_key]['class_name'] = class_name
            #out_data_list[m_key]['input']  = input.data.cpu().numpy() 
            out_data_list[m_key]['output_g'] = output_grad[0].data.cpu().numpy() 
            

        def w_hook(module, input, output):
            name = ''
            for key, val in names.items():
                if val == module:
                    name = key
            # <class 'torch.nn.modules.conv.Conv2d'>
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            if class_name in ['Conv2d','BatchNorm2d','Linear']:
                module_idx = len(out_data_list)
                m_key = module_idx + 1
                out_data_list[m_key] = OrderedDict()
                out_data_list[m_key]['name'] = name
                out_data_list[m_key]['class_name'] = class_name
                out_data_list[m_key]['weight'] = module.weight.data.cpu().numpy()
                if module.bias is not None:
                    out_data_list[m_key]['bias'] = module.bias.data.cpu().numpy()
                else:
                    out_data_list[m_key]['bias'] = None

        def r_hook(module, input, output):
            name = ''
            for key, val in names.items():
                #print('val of names 0{}0'.format(val))
                #print('module 0{}0'.format(module))
                #if '{}'.format(val).strip() == '{}'.format(module).strip():
                if val == module:
                    name = key
            # <class 'torch.nn.modules.conv.Conv2d'>
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            #print('class name: {}'.format(class_name))
            if class_name in ['aSparseFilter']:
                #print(class_name)
                module_idx = len(out_data_list)
                m_key = module_idx + 1
                out_data_list[m_key] = OrderedDict()
                out_data_list[m_key]['name'] = name
                out_data_list[m_key]['class_name'] = class_name
                out_data_list[m_key]['zero_ratio'] = module.running_zero_ratio.data.cpu()

           
        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == model):
            if type == 'a':
                hooks.append(module.register_forward_hook(a_hook))
            elif type == 'g':
                hooks.append(module.register_backward_hook(g_hook))
            elif type == 'w':
                hooks.append(module.register_forward_hook(w_hook))
            else:
                hooks.append(module.register_forward_hook(r_hook))

    # Names are stored in parent and path+name is unique not the name
    names = get_names_dict(model)
    #print(names)
    hooks = []

    # register hook
    model.apply(register_hook)

    return hooks

