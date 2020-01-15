from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from .summary import get_names_dict
from .logger import Logger
from .plot import *

import matplotlib
matplotlib.use('Tkagg') #otherwise Linux server will crash using matplotlib
import matplotlib.pyplot as plt

class Hooker:
    def __init__(self, model, vtype=None, store_list=OrderedDict()):
        '''
        store_list(OrderedDict()):
            -OrderedDict():record every layer
                -name
                -class_name
                -output
        '''
        self.model = model
        self.vtype = vtype
        self.store_list = store_list
        self.hooks = []
        self.hook_insert()

        self.name_list = []
        self.mean = []
        self.std = []
        self.vmax = []
        self.vmin = []
        self.p25 = []
        self.p50 = []
        self.p75 = []
        self.params = None

    def hook_insert(self): #model, vtype='a', layer_names=None, out_data_list=None):
        """
        Insert hooks into pytorch model
        """

        def register_hook(module): 
            """
            register_hook func is to insert hook
            check module with names dict and create named hook to insert
            """
            def create_hook(hook_name,hook_type):
                def new_hook(module, input, output):
                    name = hook_name
                    '''
                    for key, val in names.items():
                        #print(val)
                        #print(module)
                        if val == module:
                            name = key
                    '''
                    # <class 'torch.nn.modules.conv.Conv2d'>
                    class_name = str(module.__class__).split('.')[-1].split("'")[0]
                    if hook_type == 'w':
                        if hasattr(module,'weight'):
                            module_idx = len(self.store_list)
                            m_key = module_idx + 1
                            self.store_list[m_key] = OrderedDict()
                            self.store_list[m_key]['name'] = name
                            self.store_list[m_key]['class_name'] = class_name
                            self.store_list[m_key]['device'] = str(module.weight.device)
                            self.store_list[m_key]['raw_data'] = module.weight.data.cpu().numpy()
                            if module.bias is not None:
                                self.store_list[m_key]['bias'] = module.bias.data.cpu().numpy()
                            else:
                                self.store_list[m_key]['bias'] = None
                    else:
                        module_idx = len(self.store_list)
                        m_key = module_idx + 1
                        self.store_list[m_key] = OrderedDict()
                        self.store_list[m_key]['name'] = name
                        self.store_list[m_key]['class_name'] = class_name
                        self.store_list[m_key]['device'] = str(output.device)
                        self.store_list[m_key]['raw_data'] = output.data.cpu().numpy()

                return new_hook
            '''
            def g_hook(module, input_grad, output_grad):
                name = ''
                for key, val in names.items():
                    if val == module:
                        name = key
                # <class 'torch.nn.modules.conv.Conv2d'>
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(self.store_list)
                m_key = module_idx + 1
                self.store_list[m_key] = OrderedDict()
                self.store_list[m_key]['name'] = name
                self.store_list[m_key]['class_name'] = class_name
                #self.store_list[m_key]['input']  = input.data.cpu().numpy() 
                self.store_list[m_key]['output'] = output_grad[0].data.cpu().numpy() 
                
            def w_hook(module, input, output):
                name = ''
                for key, val in names.items():
                    if val == module:
                        name = key
                # <class 'torch.nn.modules.conv.Conv2d'>
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                if class_name in ['Conv2d','BatchNorm2d','Linear']:
                    module_idx = len(self.store_list)
                    m_key = module_idx + 1
                    self.store_list[m_key] = OrderedDict()
                    self.store_list[m_key]['name'] = name
                    self.store_list[m_key]['class_name'] = class_name
                    self.store_list[m_key]['output'] = module.weight.data.cpu().numpy()
                    if module.bias is not None:
                        self.store_list[m_key]['bias'] = module.bias.data.cpu().numpy()
                    else:
                        self.store_list[m_key]['bias'] = None
            '''

            if not isinstance(module, nn.Sequential) and \
                    not isinstance(module, nn.ModuleList) and \
                    not (module == self.model):
                for key, val in names.items():
                    if val == module:
                        hook_name = key
                if self.vtype == 'a':
                    self.hooks.append(module.register_forward_hook(create_hook(hook_name,self.vtype)))
                elif self.vtype == 'g':
                    self.hooks.append(module.register_backward_hook(create_hook(hook_name,self.vtype)))
                elif self.vtype == 'w':
                    self.hooks.append(module.register_forward_hook(create_hook(hook_name,self.vtype)))
                else:
                    raise ValueError('vtype error: must be a,g,w')

        # Names are stored in parent and path+name is unique not the name
        names = get_names_dict(self.model)
        #print(names)
        #hooks = []
        self.name_list = []
        self.mean = []
        self.std = []
        self.vmax = []
        self.vmin = []
        self.p25 = []
        self.p50 = []
        self.p75 = []
        self.params = None
        self.store_list = OrderedDict()
        self.hooks = []

        # register hook
        self.model.apply(register_hook)
        #print("Hooks Registered")

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        #print("Complete remove")

    def save(self, ctype = [], output_path = '', resume = False):
        '''
        self.name_list = []
        self.mean = []
        self.std = []
        self.vmax = []
        self.vmin = []
        self.p25 = []
        self.p50 = []
        self.p75 = []
        '''

        def append_data(ctype,output_data):
            if ctype == 'mean':
                self.mean.append(output_data.mean())
            elif ctype == 'std':
                self.std.append(output_data.std())
            elif ctype == 'max':
                self.vmax.append(output_data.max())
            elif ctype == 'min':
                self.vmin.append(output_data.min())
            elif ctype == 'p25':
                self.p25.append(np.percentile(output_data, 25))
            elif ctype == 'p50':
                self.p50.append(np.percentile(output_data, 50))
            elif ctype == 'p75':
                self.p75.append(np.percentile(output_data, 75))

        clist = {
            'mean':self.mean,
            'std':self.std,
            'max':self.vmax,
            'min':self.vmin,
            'p25':self.p25,
            'p50':self.p50,
            'p75':self.p75
        }   

        if len(ctype) == 0:
            print("Will save all raw data")
            for i in range(1, len(self.store_list)+1):
                data = self.store_list[i]
                if data['class_name'] in ['Conv2d','ReLU','BatchNorm2d','Linear']:
                    #print("debug",data['name'])
                    with h5py.File(output_path, 'a') as h:
                        h.create_dataset(data['name']+data['device'], data=data['raw_data'])
                    if self.vtype == 'w':
                        if data['bias'] is not None:
                            with h5py.File(output_path, 'a') as h:
                                h.create_dataset(data['name']+data['device']+'_b', data=data['bias'])
        else:
            for i in range(1, len(self.store_list)+1):
                data = self.store_list[i]
                if data['class_name'] in ['Conv2d','ReLU','BatchNorm2d','Linear']:
                    self.name_list.append(data['name']) 
                    output_data = data['output']

                    for j in range(len(ctype)):
                        append_data(ctype[j],output_data)                  
            self.params = dict([(self.name_list[i],1) for i in range(len(self.name_list))]) #dict:{name_list[0]:1, name_list[1]:1 , ......}
            
            for i in range(len(ctype)):
                logger = Logger(path = output_path, title = self.vtype + '_' + ctype[i], params = self.params, resume = resume)
                logger.save(clist[ctype[i]])

        #self.store_list = []
    
    def plot_hist(self, name, output_path = '', epoch = None, plot_range = None):
        # param name is name of layer defined by user or pytorch itself. In ResNet example, name is like: conv1,bn1,relu,layers.0.0.conv1,layers.0.0.bn1,...
        # range is a tuple
        if epoch == None:
            raise ValueError("Missing param: epoch")

        found = 0
        #for layer in self.store_list:
            #print(layer['name'])
            #if layer['name'] == name:
                #output_data = layer['output']
        for i in range(1, len(self.store_list)+1):
            data = self.store_list[i]
            if data['name'] == name:
                output_data = data['output']
                found = 1
                break
        
        if found == 0:
            raise ValueError("Fail to find this name, layer's name given is wrong")

        output_flatten = output_data.flatten()
        print('data numbers:',len(output_flatten))
        plt.hist(output_flatten, bins = 100, range = plot_range )#density = True
        hist_path = output_path + "/" + self.vtype + "_" + str(epoch) +"_" + name + ".png"
        plt.savefig(hist_path)
        plt.clf()

    def plot_curve(self, file_path):
        # run to plot w a g r curve pictures of some specific layers
        # and cross layer line of w a g r of two specific epoch
        # and distribute of w a g of one specific epoch and layer

        #file_path  = "/home/lijiajie16/fix_train_cnn_with_pytorch/checkpoint/resnet/test_hooker/log_20190522_23-35-54/"

        if self.vtype == 'a':
            # params of a curve plot
            file_list  = ['a_max', 'a_p75', 'a_p50', 'a_p25', 'a_min']
            curve_num  = len(file_list)
            mean = 'a_mean'
            std  = 'a_std'
            conv_index_list = [0,0,0,0,0]
            bn_index_list = [1,1,1,1,1]
            relu_index_list = [2,2,2,2,2]
            l02bn1_index_list = [16,16,16,16,16]
            l11bn2_index_list = [33,33,33,33,33]
            l20bnd_index_list = [42,42,42,42,42]
            l21bn1_index_list = [50,50,50,50,50]
            fc_index_list = [61,61,61,61,61,61]
            #CurvePlot(curve_num, file_path, file_list, conv_index_list, 'a_conv')
            CurvePlot(curve_num, file_path, file_list, conv_index_list,   'a_conv_vs_m' , mean, std)
            CurvePlot(curve_num, file_path, file_list,   bn_index_list,   'a_bn_vs_m'   , mean, std)
            CurvePlot(curve_num, file_path, file_list, relu_index_list,   'a_relu_vs_m' , mean, std)
            CurvePlot(curve_num, file_path, file_list, l02bn1_index_list, 'a_02bn1_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list, l11bn2_index_list, 'a_11bn2_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list, l20bnd_index_list, 'a_20bnd_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list, l21bn1_index_list, 'a_21bn1_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list,   fc_index_list,   'a_fc_vs_m'   , mean, std)

        if self.vtype == 'g':
            # params of g curve plot
            file_list  = ['g_max', 'g_p75', 'g_p50', 'g_p25', 'g_min']
            curve_num  = len(file_list)
            mean = 'g_mean'
            std  = 'g_std'
            conv_index_list = [61,61,61,61,61,61]
            bn_index_list = [60,60,60,60,60,60]
            relu_index_list = [59,59,59,59,59,59]
            l02re1_index_list = [44,44,44,44,44]
            l11re2_index_list = [27,27,27,27,27]
            l20cod_index_list = [20,20,20,20,20]
            l21re1_index_list = [10,10,10,10,10]
            fc_index_list = [0,0,0,0,0]
            #CurvePlot(curve_num, file_path, file_list, conv_index_list, 'a_conv')
            CurvePlot(curve_num, file_path, file_list, conv_index_list,   'g_conv_vs_m' , mean, std)
            CurvePlot(curve_num, file_path, file_list,   bn_index_list,   'g_bn_vs_m'   , mean, std)
            CurvePlot(curve_num, file_path, file_list, relu_index_list,   'g_relu_vs_m' , mean, std)
            CurvePlot(curve_num, file_path, file_list, l02re1_index_list, 'g_02relu1_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list, l11re2_index_list, 'g_11relu2_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list, l20cod_index_list, 'g_20convd_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list, l21re1_index_list, 'g_21relu1_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list,   fc_index_list,   'g_fc_vs_m'   , mean, std)

        if self.vtype == 'w':
            # params of w curve plot
            file_list  = ['w_max', 'w_p75', 'w_p50', 'w_p25', 'w_min']
            curve_num  = len(file_list)
            mean = 'w_mean'
            std  = 'w_std'
            conv_index_list = [0,0,0,0,0]
            bn_index_list = [1,1,1,1,1]
            l02co1_index_list = [10,10,10,10,10]
            l11bn2_index_list = [23,23,23,23,23]
            l20cod_index_list = [28,28,28,28,28]
            l21co2_index_list = [36,36,36,36,36]
            fc_index_list = [42,42,42,42,42]
            #CurvePlot(curve_num, file_path, file_list, conv_index_list, 'a_conv')
            CurvePlot(curve_num, file_path, file_list, conv_index_list,   'w_conv_vs_m' , mean, std)
            CurvePlot(curve_num, file_path, file_list,   bn_index_list,   'w_bn_vs_m'   , mean, std)
            CurvePlot(curve_num, file_path, file_list, l02co1_index_list, 'w_02conv1_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list, l11bn2_index_list, 'w_11bn2_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list, l20cod_index_list, 'w_20convd_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list, l21co2_index_list, 'w_21conv2_vs_m', mean, std)
            CurvePlot(curve_num, file_path, file_list,   fc_index_list,   'w_fc_vs_m'   , mean, std)

    def plot_line(self, file_path, epoch):
        line_index = epoch + 1
        if self.vtype == 'a':
            file_list  = ['a_max', 'a_p75', 'a_p50', 'a_p25', 'a_min']
            line_num  = len(file_list)
            LinePlot(line_num, file_path, file_list, line_index, 'cross_layer_a'+"_"+str(epoch))

        if self.vtype == 'g':
            file_list  = ['g_max', 'g_p75', 'g_p50', 'g_p25', 'g_min']
            line_num  = len(file_list)
            LinePlot(line_num, file_path, file_list, line_index, 'cross_layer_g'+"_"+str(epoch))

        if self.vtype == 'w':
            file_list  = ['w_max', 'w_p75', 'w_p50', 'w_p25', 'w_min']
            line_num  = len(file_list)
            LinePlot(line_num, file_path, file_list, line_index, 'cross_layer_w'+"_"+str(epoch))
