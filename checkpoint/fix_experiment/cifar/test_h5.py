import numpy as np
import h5py
import os 
import sys

dirname = 'log_quantize_check'
typename = 'e'
epoch = 90
layername = 'layers.0.2.bn2'
filename = os.path.join(dirname,'checkpoint_'+typename+'_'+str(epoch)+'.h5')

if not os.path.exists(filename):
    print(filename)
    exit("wrong filename")

f = h5py.File(filename,'r')
keys = f.keys()
for layer_name in keys:
    layer_data = f[layer_name][:]
    layer_name = layer_name[0:-6]
    if layer_name == layername:
        print(layer_name)
        shape = layer_data.shape
        print('layer_shape',shape)
        print('data range',layer_data.min(), layer_data.max())
        min_list = np.sort(layer_data.min(-1).min(-1).min(-1))
        max_list = np.sort(layer_data.max(-1).max(-1).max(-1))
        abs_list = np.sort(np.abs(layer_data).max(-1).max(-1).max(-1))
        print('n group range', min_list, max_list)
        print('n group abs',abs_list[0],abs_list[-1])
        print('list len', min_list.shape, abs_list.shape)
        print('section 1/8', np.sum(abs_list<=0.125*abs_list.max()))
        print('section 1/4', np.sum(abs_list<=0.25*abs_list.max()))
        print('section 1/2', np.sum(abs_list<=0.5*abs_list.max()))
