import torch
import numpy as np
import h5py
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from quantize_functions import Quantize_A

prefix = '../../../checkpoint/fix_experiment/imgnet/log_quantize_20200926_12-32-39/checkpoint_'
epochnum = '28'
mannum = 4

def plotexp(data, epoch, man):
    f = h5py.File(prefix+data+'_'+epoch+'.h5','r')
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
            if count % 8 == 0:
                print('  ',layer_name)
                shape = layer_data.shape
                print('    shape',shape)
                print('    range',layer_data.min(), layer_data.max())
                x_in = torch.Tensor(layer_data)

                q_xf = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          0,                                                  
                                          False,                                                 
                                          False,                                                  
                                          False,                                                  
                                          'complex',                                                 
                                          'progress')
                q_xn = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          1,                                                  
                                          False,                                                 
                                          False,                                                  
                                          False,                                                  
                                          'complex',                                                 
                                          'progress')
                q_xc = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          2,                                                  
                                          False,                                                 
                                          False,                                                  
                                          False,                                                   
                                          'complex',                                                 
                                          'progress')
                q_xnc = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          3,                                                  
                                          False,                                                 
                                          False,                                                  
                                          False,                                                   
                                          'complex',                                                 
                                          'progress')
                ef = np.average(np.abs((q_xf - x_in)/(x_in+1e-20)))
                en = np.average(np.abs((q_xn - x_in)/(x_in+1e-20)))
                ec = np.average(np.abs((q_xc - x_in)/(x_in+1e-20)))
                enc = np.average(np.abs((q_xnc - x_in)/(x_in+1e-20)))
                print('average: ',layer_name, ef,en,ec,enc)

                fresults = np.append(fresults,ef)
                nresults = np.append(nresults,en)
                cresults = np.append(cresults,ec)
                ncresults = np.append(ncresults,enc)
            count += 1

    print(fresults)
    print(fresults.shape)
    nn = len(fresults)
    x = 1 + np.arange(0,nn)

    plt.figure()
    plt.tick_params(labelsize=10)
    ax=plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    f, = plt.plot(x[1:], fresults[1:], alpha = 0.25, linewidth=3)
    n, = plt.plot(x[1:], nresults[1:], alpha = 0.5,  linewidth=3)
    c, = plt.plot(x[1:], cresults[1:], alpha = 0.75, linewidth=3)
    nc, = plt.plot(x[1:],ncresults[1:],alpha = 1,    linewidth=3)
    plt.legend([f,n,c,nc],['E_x=0', 'E_x=1', 'E_x=2','E_x=3'], prop={'size':12})
    plt.xlabel('layer id',{'size':15})
    plt.ylabel('average relative error',{'size':15})

    plt.savefig(data+'_'+epoch+'exp'+'.png')
    
    
def plotnc(data, epoch, man):
    f = h5py.File(prefix+data+'_'+epoch+'.h5','r')
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
            if count % 8 == 0:
                print('  ',layer_name)
                shape = layer_data.shape
                print('    shape',shape)
                print('    range',layer_data.min(), layer_data.max())
                x_in = torch.Tensor(layer_data)

                q_xf = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          0,                                                  
                                          False,                                                 
                                          False,                                                  
                                          False,                                                  
                                          'complex',                                                 
                                          'progress')
                q_xn = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          0,                                                  
                                          False,                                                 
                                          False,                                                  
                                          'n',                                                  
                                          'complex',                                                 
                                          'progress')
                q_xc = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          0,                                                  
                                          False,                                                 
                                          False,                                                  
                                          'c',                                                   
                                          'complex',                                                 
                                          'progress')
                q_xnc = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          0,                                                  
                                          False,                                                 
                                          False,                                                  
                                          'nc',                                                   
                                          'complex',                                                 
                                          'progress')
                ef = np.average(np.abs((q_xf - x_in)/(x_in+1e-20)))
                en = np.average(np.abs((q_xn - x_in)/(x_in+1e-20)))
                ec = np.average(np.abs((q_xc - x_in)/(x_in+1e-20)))
                enc = np.average(np.abs((q_xnc - x_in)/(x_in+1e-20)))
                print('average: ',layer_name, ef,en,ec,enc)

                fresults = np.append(fresults,ef)
                nresults = np.append(nresults,en)
                cresults = np.append(cresults,ec)
                ncresults = np.append(ncresults,enc)
            count += 1

    print(fresults)
    print(fresults.shape)
    nn = len(fresults)
    x = 1+np.arange(0,nn)

    plt.figure()
    plt.tick_params(labelsize=10)
    ax=plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    f, = plt.plot(x[1:], fresults[1:], alpha = 0.25, linewidth=3)
    n, = plt.plot(x[1:], nresults[1:], alpha = 0.5,  linewidth=3)
    c, = plt.plot(x[1:], cresults[1:], alpha = 0.75, linewidth=3)
    nc, = plt.plot(x[1:],ncresults[1:],alpha = 1,    linewidth=3)
    plt.legend([f,n,c,nc],['1 group', 'N groups', 'C groups','NxC groups'], prop={'size':12})
    plt.xlabel('layer id',{'size':15})
    plt.ylabel('average relative error',{'size':15})

    plt.savefig(data+'_'+epoch+'nc'+'.png')


def plotncexp(data, epoch, man):
    f = h5py.File(prefix+data+'_'+epoch+'.h5','r')
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
            if count % 8 == 0:
                print('  ',layer_name)
                shape = layer_data.shape
                print('    shape',shape)
                print('    range',layer_data.min(), layer_data.max())
                x_in = torch.Tensor(layer_data)

                q_xf = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          0,                                                  
                                          False,                                                 
                                          False,                                                  
                                          'nc',                                                  
                                          'complex',                                                 
                                          'progress')
                q_xn = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          1,                                                  
                                          False,                                                 
                                          False,                                                  
                                          'nc',                                                  
                                          'complex',                                                 
                                          'progress')
                q_xc = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          2,                                                  
                                          False,                                                 
                                          False,                                                  
                                          'nc',                                                   
                                          'complex',                                                 
                                          'progress')
                q_xnc = Quantize_A.apply(x_in,                                                 
                                          man,                                                  
                                          3,                                                  
                                          False,                                                 
                                          False,                                                  
                                          'nc',                                                   
                                          'complex',                                                 
                                          'progress')
                ef = np.average(np.abs((q_xf - x_in)/(x_in+1e-20)))
                en = np.average(np.abs((q_xn - x_in)/(x_in+1e-20)))
                ec = np.average(np.abs((q_xc - x_in)/(x_in+1e-20)))
                enc = np.average(np.abs((q_xnc - x_in)/(x_in+1e-20)))
                print('average: ',layer_name, ef,en,ec,enc)

                fresults = np.append(fresults,ef)
                nresults = np.append(nresults,en)
                cresults = np.append(cresults,ec)
                ncresults = np.append(ncresults,enc)
            count += 1

    print(fresults)
    print(fresults.shape)
    nn = len(fresults)
    x = np.arange(0,nn)

    plt.figure()
    plt.tick_params(labelsize=10)
    ax=plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    f, = plt.plot(x[1:], fresults[1:], alpha = 0.25, linewidth=3)
    n, = plt.plot(x[1:], nresults[1:], alpha = 0.5,  linewidth=3)
    c, = plt.plot(x[1:], cresults[1:], alpha = 0.75, linewidth=3)
    nc, = plt.plot(x[1:],ncresults[1:],alpha = 1,    linewidth=3)
    plt.legend([f,n,c,nc],['E_x=0', 'E_x=1', 'E_x=2','E_x=3'], prop={'size':12})
    plt.xlabel('layer id',{'size':15})
    plt.ylabel('average relative error',{'size':15})

    plt.savefig(data+'_'+epoch+'ncexp'+'.png')

    
plotnc('e',epochnum, mannum)
plotexp('e', epochnum, mannum)
plotncexp('e',epochnum, mannum)

plotnc('a',epochnum, mannum)
plotexp('a', epochnum, mannum)
plotncexp('a',epochnum, mannum)

plotnc('w',epochnum, mannum)
plotexp('w', epochnum, mannum)
plotncexp('w',epochnum, mannum)
