import numpy as np
import matplotlib.pyplot as plt
import h5py
import os 
import sys

groupdim = 0
dirname = 'log_quantize_check'
typename = 'a'
epoch = 15
blocknum = 'layers.2.0.'
layernum = 2
layername = (blocknum+'relu'+str(layernum)) if typename=='a' else (blocknum+'conv'+str(layernum))
filename = os.path.join(dirname,'checkpoint_'+typename+'_'+str(epoch)+'.h5')

if not os.path.exists(filename):
    print(filename)
    exit("wrong filename")

f = h5py.File(filename,'r')
keys = f.keys()
for layer_name in keys:
    layer_data = f[layer_name][:]
    layer_name = layer_name[0:-8] if layer_name[-1]=='b' else layer_name[0:-6]
    #print(layer_name)
    if layer_name == layername:
        print(layer_name)
        shape = layer_data.shape
        print('layer_shape',shape)
        min_list = np.sort(layer_data.min(-1).min(-1).min(groupdim))
        max_list = np.sort(layer_data.max(-1).max(-1).max(groupdim))
        abs_list = np.sort(np.abs(layer_data).max(-1).max(-1).max(groupdim))
        print('data range',layer_data.min(), layer_data.max())
        print('group abs range',abs_list[0],abs_list[-1])
        print('num of groups', abs_list.shape)
        print('section 1/32', np.sum(abs_list<=0.03125*abs_list.max()))
        print('section 1/16', np.sum(abs_list<=0.0625*abs_list.max()))
        print('section 1/8', np.sum(abs_list<=0.125*abs_list.max()))
        print('section 1/4', np.sum(abs_list<=0.25*abs_list.max()))
        print('section 1/2', np.sum(abs_list<=0.5*abs_list.max()))
