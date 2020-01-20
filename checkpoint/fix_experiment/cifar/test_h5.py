import numpy as np
import h5py

dir = 'log_quantize_check-d/'
#dir = 'log_quantize_check/'

#filename = 'checkpoint_w_0.h5'
#filename = 'checkpoint_a_0.h5'
filename = 'checkpoint_e_0.h5'
#filename = 'checkpoint_g_0.h5'

#filename = 'checkpoint_w_2.h5'
#filename = 'checkpoint_a_2.h5'
#filename = 'checkpoint_e_2.h5'
#filename = 'checkpoint_g_2.h5'

f = h5py.File(dir+filename)
keys = f.keys()
for layer_name in keys:
    layer_data = f[layer_name]
    layer_shape = layer_data.shape
    if len(layer_shape) == 4:
        print(layer_name)
        print('  layer_shape',layer_data.shape)
        print('  layer_value',layer_data[0,1,0:2,0:2])
    #print(layer_name)
    #print('  layer_shape',layer_data.shape)

