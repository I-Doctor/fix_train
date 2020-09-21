######################################################################
# (c) Copyright EFC of NICS, Tsinghua University. All rights reserved.
#  Author: Kai Zhong
#  Email : zhongk19@mails.tsinghua.edu.cn
#  
#  Create Date : 2020.08.26
#  File Name   : plot_h5.py
#  Description : read the inner data of training with h5 type
#                and plot distribute figure 
#  Dependencies: 
######################################################################

import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



groupdim = 0
'''
dirname = 'log_quantize_check'
typename = 'a'
epoch = 15
blocknum = 'layers.2.0.'
layernum = 2
layername = (blocknum+'relu'+str(layernum)) if typename=='a' else (blocknum+'conv'+str(layernum))
filename = os.path.join(dirname,'checkpoint_'+typename+'_'+str(epoch)+'.h5')
'''



def plot_file(filename, output_path, config_names, count):
    splitname = filename.split('_')
    data_type = splitname[-2]
    epoch = splitname[-1]
    f = h5py.File(filename,'r')
    keys = f.keys()
    for layer_name in keys:
        layer_data = f[layer_name][:]
        layer_name = layer_name[0:-8] if layer_name[-1]=='b' else layer_name[0:-6]
        print('  ',layer_name)
        shape = layer_data.shape
        print('    shape',shape)
        print('    range',layer_data.min(), layer_data.max())

        data_flatten = layer_data.flatten()
        print('    data numbers:',len(data_flatten))
        plt.hist(data_flatten, bins = 100, range = plot_range)#density = True
        hist_path = os.path.join(output_path, (config_names[count]+"_"+name+"_"+str(epoch)+".png"))
        plt.savefig(hist_path)
        plt.clf()



def process_file(filename, args):
    splitname = filename.split('_')
    data_type = splitname[-2]
    epoch = splitname[-1]
    f = h5py.File(filename,'r')
    keys = f.keys()
    for layer_name in keys:
        layer_data = f[layer_name][:]
        layer_name = layer_name[0:-8] if layer_name[-1]=='b' else layer_name[0:-6]
        '''
        if layer_name == layername:
            #print(layer_name)
            shape = layer_data.shape
            #print('layer_shape',shape)
            min_list = np.sort(layer_data.min(-1).min(-1).min(groupdim))
            max_list = np.sort(layer_data.max(-1).max(-1).max(groupdim))
            abs_list = np.sort(np.abs(layer_data).max(-1).max(-1).max(groupdim))
            print('    data range',layer_data.min(), layer_data.max())
            print('    group abs range',abs_list[0],abs_list[-1])
            print('    num of groups', abs_list.shape)
            print('section 1/32', np.sum(abs_list<=0.03125*abs_list.max()))
            print('section 1/16', np.sum(abs_list<=0.0625*abs_list.max()))
            print('section 1/8', np.sum(abs_list<=0.125*abs_list.max()))
            print('section 1/4', np.sum(abs_list<=0.25*abs_list.max()))
            print('section 1/2', np.sum(abs_list<=0.5*abs_list.max()))
        '''
        print('  ',layer_name)
        shape = layer_data.shape
        print('    shape',shape)
        print('    range',layer_data.min(), layer_data.max())



def main(argv):

    print(argparse)
    print(type(argparse))

    parser = argparse.ArgumentParser()

    # required arguments: 
    parser.add_argument(
        "plot_type",
        help = "what type of figures are you going to plot.\n\
                supported: dist log_dist cross_layer cross_epoch"
    )
    parser.add_argument(
        "output_dir",
        help = "the name of output dir to store the results."
    )
    parser.add_argument(
        "--statistic_names",
        nargs='+',
        default=None,
        help = "what statistic values are you going to plot or compare.\n        \
                supported: mean_var max_min five"
    )
    parser.add_argument(
        "--data_types",
        help = "what types of data are you going to show.\n        \
                example: all a w e g"
    )
    parser.add_argument(
        "--file_range",
        nargs='+',
        help = "the date range of input file to read the results."
    )

    args = parser.parse_args()
    #print(args.file_range)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    elif not args.output_dir.endswith('test_out'):
        exit('output dir exists')

    dirlist = os.listdir('./')
    print(dirlist)



    count = 0
    for dirname in dirlist:
        if dirname.startswith('log_quantize'):
            date = dirname[13:21]
            #print(date)
            if date >= args.file_range[0] and date <= args.file_range[1]:
                #print('date ok')
                filelist = os.listdir(dirname)
                #print(filelist)
                for filename in filelist:
                    if filename.endswith('.h5'):
                        filepath = os.path.join(dirname,filename)
                        print(filepath)
                        if not os.path.exists(filepath):
                            print(filepath)
                            exit("wrong filename")
                        if args.data_types == 'all':
                            plot_file(filepath, args.output_dir, config_names, count):
                        else if args.data_types in ['a','w','e','g']:
                            if filepath[11] == args.data_types: 
                                plot_file(filepath, args.output_dir, config_names, count):
                        else:
                            exit("wrong type")
                        count += 1



if __name__ == '__main__':

    main(sys.argv)

