import h5py
import torch 
import numpy as np
import matplotlib.pyplot as plt
from quantize_functions import Quantize_A

f = h5py.File('../../../checkpoint/fix_experiment/imgnet/log_quantize_20200926_12-32-39/checkpoint_e_56.h5','r')
keys = f.keys()
fresults = np.array([])
nresults = np.array([])
cresults = np.array([])
ncresults = np.array([])
count = 0
for layer_name in keys:
    layer_data = f[layer_name][:]
    layer_name = layer_name[0:-8] if layer_name[-1]=='b' else layer_name[0:-6]
    if layer_name[0:-1].endswith('conv'):
        print('  ',layer_name)
        shape = layer_data.shape
        print('    shape',shape)
        print('    range',layer_data.min(), layer_data.max())
        x_in = torch.Tensor(layer_data)

        q_xf = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  0,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        q_xn = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  1,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        q_xc = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  2,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        q_xnc = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  3,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        if count % 8 == 0:
            ef = np.average(np.abs((q_xf - x_in)/(x_in+1e-20)))
            en = np.average(np.abs((q_xn - x_in)/(x_in+1e-20)))
            ec = np.average(np.abs((q_xc - x_in)/(x_in+1e-20)))
            enc = np.average(np.abs((q_xnc - x_in)/(x_in+1e-20)))
            print("results:", ef,en,ec,enc)
            fresults = np.append(fresults,ef)
            nresults = np.append(nresults,en)
            cresults = np.append(cresults,ec)
            ncresults = np.append(ncresults,enc)
        count += 1
        
print(fresults)
print(fresults.shape)
nn = len(fresults)
x = 1+ np.arange(0,nn)

plt.figure()
f, = plt.plot(x, fresults,linewidth=2)
n, = plt.plot(x, nresults,linewidth=2)
c, = plt.plot(x,cresults,linewidth=2)
nc, = plt.plot(x,ncresults,linewidth=2)
plt.legend([f,n,c,nc],['E_x=0', 'E_x=1', 'E_x=2','E_x=3'])
plt.xlabel('layer id')
plt.ylabel('average relative error')

plt.savefig('e3.png')


f = h5py.File('../../../checkpoint/fix_experiment/imgnet/log_quantize_20200926_12-32-39/checkpoint_a_56.h5','r')
keys = f.keys()
fresults = np.array([])
nresults = np.array([])
cresults = np.array([])
ncresults = np.array([])
count = 0
for layer_name in keys:
    layer_data = f[layer_name][:]
    layer_name = layer_name[0:-8] if layer_name[-1]=='b' else layer_name[0:-6]
    if layer_name[0:-1].endswith('conv'):
        print('  ',layer_name)
        shape = layer_data.shape
        print('    shape',shape)
        print('    range',layer_data.min(), layer_data.max())
        x_in = torch.Tensor(layer_data)

        q_xf = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  0,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        q_xn = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  1,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        q_xc = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  2,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        q_xnc = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  3,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        if count % 8 == 0:
            ef = np.average(np.abs((q_xf - x_in)/(x_in+1e-20)))
            en = np.average(np.abs((q_xn - x_in)/(x_in+1e-20)))
            ec = np.average(np.abs((q_xc - x_in)/(x_in+1e-20)))
            enc = np.average(np.abs((q_xnc - x_in)/(x_in+1e-20)))
            print("results:", ef,en,ec,enc)
            fresults = np.append(fresults,ef)
            nresults = np.append(nresults,en)
            cresults = np.append(cresults,ec)
            ncresults = np.append(ncresults,enc)
        count += 1
        
print(fresults)
print(fresults.shape)
nn = len(fresults)
x = 1+ np.arange(0,nn)

plt.figure()
f, = plt.plot(x, fresults,linewidth=2)
n, = plt.plot(x, nresults,linewidth=2)
c, = plt.plot(x,cresults,linewidth=2)
nc, = plt.plot(x,ncresults,linewidth=2)
plt.legend([f,n,c,nc],['E_x=0', 'E_x=1', 'E_x=2','E_x=3'])
plt.xlabel('layer id')
plt.ylabel('average relative error')

plt.savefig('a3.png')


f = h5py.File('../../../checkpoint/fix_experiment/imgnet/log_quantize_20200926_12-32-39/checkpoint_w_56.h5','r')
keys = f.keys()
fresults = np.array([])
nresults = np.array([])
cresults = np.array([])
ncresults = np.array([])
count = 0
for layer_name in keys:
    layer_data = f[layer_name][:]
    layer_name = layer_name[0:-8] if layer_name[-1]=='b' else layer_name[0:-6]
    if layer_name[0:-1].endswith('conv'):
        print('  ',layer_name)
        shape = layer_data.shape
        print('    shape',shape)
        print('    range',layer_data.min(), layer_data.max())
        x_in = torch.Tensor(layer_data)

        q_xf = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  0,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        q_xn = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  1,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        q_xc = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  2,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        q_xnc = Quantize_A.apply(x_in,                                                 
                                  4,                                                  
                                  3,                                                  
                                  False,                                                 
                                  False,                                                  
                                  'nc',                                                  
                                  'complex',                                                 
                                  'progress')
        if count % 8 == 0:
            ef = np.average(np.abs((q_xf - x_in)/(x_in+1e-20)))
            en = np.average(np.abs((q_xn - x_in)/(x_in+1e-20)))
            ec = np.average(np.abs((q_xc - x_in)/(x_in+1e-20)))
            enc = np.average(np.abs((q_xnc - x_in)/(x_in+1e-20)))
            print("results:", ef,en,ec,enc)
            fresults = np.append(fresults,ef)
            nresults = np.append(nresults,en)
            cresults = np.append(cresults,ec)
            ncresults = np.append(ncresults,enc)
        count += 1
        
print(fresults)
print(fresults.shape)
nn = len(fresults)
x = 1+ np.arange(0,nn)

plt.figure()
f, = plt.plot(x, fresults,linewidth=2)
n, = plt.plot(x, nresults,linewidth=2)
c, = plt.plot(x,cresults,linewidth=2)
nc, = plt.plot(x,ncresults,linewidth=2)
plt.legend([f,n,c,nc],['E_x=0', 'E_x=1', 'E_x=2','E_x=3'])
plt.xlabel('layer id')
plt.ylabel('average relative error')

plt.savefig('w3.png')

