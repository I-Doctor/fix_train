import numpy as np
import h5py

#filename = 'checkpoint_w_0.h5'
#filename = 'checkpoint_a_0.h5'
#filename = 'checkpoint_e_0.h5'
#filename = 'checkpoint_g_0.h5'

#filename = 'checkpoint_w_2.h5'
filename = 'checkpoint_a_2.h5'
#filename = 'checkpoint_e_2.h5'
#filename = 'checkpoint_g_2.h5'

f = h5py.File(filename)
keys = f.keys()
for layer_name in keys:
    layer_data = f[layer_name]
    print(layer_name)
    print('layer1_shape',layer_data.shape)
