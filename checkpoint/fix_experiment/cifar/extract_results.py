######################################################################
# (c) Copyright EFC of NICS, Tsinghua University. All rights reserved.
#  Author: Kai Zhong
#  Email : zhongk19@mails.tsinghua.edu.cn
#  
#  Create Date : 2020.08.16
#  File Name   : read_results.py
#  Description : read the config of train and test accuracy data from
#                log file and show on one screen to compare
#  Dependencies: 
######################################################################

import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def clean(s):
    ''' clean string to drop , ' : [ ] { } and set bool to T F
    '''
    return s.replace(",","").replace("'","").replace(':','').replace('[','').replace(']','').replace('{','').replace('}','').replace('True','T').replace('False','F').replace('None','N')



def check_column(configs, column_label):
    ''' check if there is already column named column_label
    '''
    if column_label in configs.columns.values.tolist():
        return True
    else:
        return False



def add_line(configs, count, wordlist, pos):
    ''' add info in one line of one file into dataframe configs
        count is the line index
        wordlist is the word list of this line
        pos=1 means first level configs and pos=3 means second
    '''
    # first level configs
    if pos == 1:
        column_label = wordlist[0]
        if check_column(configs, column_label):
            configs.loc[count,(column_label)] = clean(wordlist[2]) \
                    if column_label != 'output_dir' else wordlist[2][-17:]
        else:
            configs[column_label] = None
            configs.loc[count,(column_label)] = clean(wordlist[2]) \
                    if column_label != 'output_dir' else wordlist[2][-17:]

    # second level configs
    elif pos == 3:
        # deal with q_cfg
        if wordlist[2] == 'q_cfg':
            for i in range(4, len(wordlist)):
                if wordlist[i].endswith("':"):
                    # record label and element
                    column_label = clean(wordlist[i])
                    data_element = clean(wordlist[i+1])
                    # append element until next label
                    for j in range(i+2, len(wordlist)):
                        if wordlist[j].endswith("':"): break
                        #else: data_element += ','+clean(wordlist[j])
                        else: data_element += clean(wordlist[j])
                    if check_column(configs, column_label):
                        configs.loc[count,(column_label)] = data_element
                    else:
                        configs[column_label] = None
                        configs.loc[count,(column_label)] = data_element
        # len > 5 and not q_cfg means list configs
        elif len(wordlist) > 5:
            # record label and element
            column_label = clean(wordlist[0])+clean(wordlist[2])
            data_element = clean(wordlist[4])
            # append element until the end of this line
            for i in range(5, len(wordlist)):
                #data_element += ','+clean(wordlist[i])
                data_element += clean(wordlist[i])
            if check_column(configs, column_label):
                configs.loc[count,(column_label)] = data_element
            else:
                configs[column_label] = None
                configs.loc[count,(column_label)] = data_element
        # !len > 5 means one element configs
        else:
            column_label = clean(wordlist[0])+clean(wordlist[2])
            data_element = clean(wordlist[4])
            if check_column(configs, column_label):
                configs.loc[count,(column_label)] = data_element
            else:
                configs[column_label] = None
                configs.loc[count,(column_label)] = data_element
    else:
        print(wordlist, pos)
        exit("wrong : position")



def add_results(results, count, column_label, column_data):
    ''' add one result into results
    '''
    if not column_data.endswith('%'):
        column_data = float(column_data)
    else:
        column_data = float(column_data[:-1])
    if check_column(results, column_label):
        results.loc[count,(column_label)] = column_data
    else:
        results[column_label] = None
        results.loc[count,(column_label)] = column_data



def process_file(filepath, configs, results, count):
    ''' process one file line by line and add all configs
        and values into dataframe
    '''
    with open(filepath) as f:
        temp_epoch = 0
        train_acc  = 0
        train_loss = 0
        test_loss  = 0
        for line in f:                                  # check line by line
            wordlist = line.split()                     # split one line to a list
            # process long config lines with : at position 3
            if len(wordlist) >= 5 and wordlist[0] != 'Accuracy'\
            and wordlist[0] != 'LOG':
                if wordlist[3]==':':
                    add_line(configs, count, wordlist, 3)         # add this line to configs
            # process long config lines with : at position 1
            elif len(wordlist) >= 3 and wordlist[0] != 'gpu':
                if wordlist[1]==':': 
                    add_line(configs, count, wordlist, 1)         # add this line to configs

            # process best result
            if len(wordlist) > 1:
                # add best acc
                if wordlist[0] == 'BEST':
                    add_results(results, count, 'BestAcc', wordlist[2])
                    add_results(results, count, 'BestEpoch', wordlist[5])
                # add train loss and acc
                elif wordlist[0] == 'Epoch:':
                    train_acc  = wordlist[13][1:-1]
                    train_loss = wordlist[10][1:-1]
                # add test loss
                elif wordlist[0] == 'Test:':
                    test_loss = wordlist[7][1:-1]
                # add test acc and save all results in this epoch to results
                elif wordlist[0] == '*':
                    add_results(results, count, str(temp_epoch)+'TrainAcc', train_acc)
                    add_results(results, count, str(temp_epoch)+'TrainLoss', train_loss)
                    add_results(results, count, str(temp_epoch)+'TestLoss', test_loss)
                    add_results(results, count, str(temp_epoch)+'TestAcc', wordlist[2])
                    add_results(results, count, str(temp_epoch)+'Test5Acc', wordlist[4])
                    temp_epoch += 1

        return temp_epoch
                


def plot_file(filename, output_path, config_names, count, dist_type):
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
        if dist_type.startswith('log'):
            data_flatten = np.where(data_flatten<1e-12,1e-12,data_flatten)
            data_flatten = np.log2(np.abs(data_flatten))
            if np.min(data_flatten) < -100:
                print(np.min(data_flatten))
        #plt.hist(data_flatten, bins = 100, range = layer_data.max()-layer_data.min())#density = True
        if data_type == 'e':
            if np.min(data_flatten) < -100:
                print('e', np.min(data_flatten))
            plt.hist(data_flatten, bins = 100, log=True)
        else:
            plt.hist(data_flatten, bins = 100)
        hist_path = os.path.join(output_path, (config_names[count]+"_"+layer_name+"_"+data_type+"_"+str(epoch)+".png"))
        plt.savefig(hist_path)
        plt.clf()



def main(argv):

    #print(argparse)
    #print(type(argparse))

    parser = argparse.ArgumentParser()

    # required arguments: 
    parser.add_argument(
        "type",
        help = "what type of mission are you going to do.\n\
                supported: compare loss acc dist log_dist"
    )
    parser.add_argument(
        "output_dir",
        help = "the name of output dir to store the results."
    )
    parser.add_argument(
        "--results_name",
        help = "what results are you going to plot or compare.\n        \
                Only work for compare loss acc.               \n        \
                supported: best_acc test_acc train_acc test_loss train_loss"
    )
    parser.add_argument(
        "--data_types",
        help = "what types of data are you going to plot dist.\n        \
                Only work for dist and log_dist.              \n        \
                supported: all w a g e"
    )
    parser.add_argument(
        "--config_name",
        help = "what configs are you going to show.\n        \
                example: all bw group hard "
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

    configs = pd.DataFrame()
    print(configs)
    results = pd.DataFrame()
    print(results)
    count = 0
    epoch_num = 0

    for dir in dirlist:
        if dir.startswith('log_quantize'):
            date = dir[13:21]
            #print(date)
            if date >= args.file_range[0] and date <= args.file_range[1]:
                #print('date ok')
                filelist = os.listdir(dir)
                #print(filelist)
                for filename in filelist:
                    if filename.endswith('.log'):
                        filepath = os.path.join(dir,filename)
                        print(filepath)
                        if not os.path.exists(filepath):
                            print(filepath)
                            exit("wrong filename")
                        epoch_num = process_file(filepath, configs, results, count)
                        count += 1

    configs.to_csv(os.path.join(args.output_dir, 'all_configs.csv'))
    results.to_csv(os.path.join(args.output_dir, 'all_results.csv'))
    # check configs and generate values
    #print(count)
    #print(results)
    configs = configs.loc[:, (configs != configs.iloc[0]).any()]
    print(configs)
    TrainLoss = results[['%dTrainLoss'%i for i in range(epoch_num)]]
    TrainAcc  = results[['%dTrainAcc'%i for i in range(epoch_num)]]
    TestLoss = results[['%dTestLoss'%i for i in range(epoch_num)]]
    TestAcc  = results[['%dTestAcc'%i for i in range(epoch_num)]]
    
    values = {'train_loss': TrainLoss, 'train_acc': TrainAcc,
              'test_loss': TestLoss, 'test_acc': TestAcc}

    # output datas with args
    if args.type == 'compare':
        # compare best acc throught table
        if args.results_name == 'best_acc':
            output = pd.concat([configs, results[['BestAcc','BestEpoch']]], axis=1)
            #print(output)
            output.to_csv(os.path.join(args.output_dir,'compare_best.csv'))
            sort_output = output.sort_values('BestAcc')
            print(sort_output)
            sort_output.to_csv(os.path.join(args.output_dir,'sorted_best.csv'))

        # compare one type of curve across configs by figure
        elif args.results_name.endswith('acc') or args.results_name.endswith('loss'):
            # generate labels by list configs or all configs or someone config
            if isinstance(args.config_name, list):
                legends = configs[args.config_name[0]].str.cat(configs[args.config_name[1:]], sep='-', na_rep='*')
            elif args.config_name == 'all':
                legends = configs.iloc[:,0].str.cat(configs.iloc[:,1:-1], sep='-', na_rep='*')
                #legends = configs.applymap(str).iloc[:,0].str.cat(configs.iloc[:,1:-1].applymap(str), sep='-', na_rep='*')
                #print(legends)
            else:
                legends = configs[args.config_name]
            plot_data = values[args.results_name] 
            plot_data = pd.DataFrame(plot_data.values.T, 
                                     index   = range(epoch_num),
                                     columns = legends)
            print(plot_data)
            fig = plt.figure()
            ax  = fig.add_subplot(1,1,1)
            plot_data.plot(ax=ax, 
                           xticks = range(0,epoch_num,15),
                           kind='line')
            fig.show()
            fig.savefig(os.path.join(args.output_dir,'compare_'+args.results_name+'_'+args.config_name+'.png'))

        # compare all curves throught figures
        else:
            exit("can't compare all results of different configs")

    # plot one type of curves of each config onto figures
    elif args.type == 'loss' or args.type == 'acc':
        train_data = values['train_'+args.type] 
        test_data  = values['test_'+args.type] 
        titles = configs.iloc[:,0].str.cat(configs.iloc[:,1:-1], sep='-', na_rep='*')
        for i in range(count):
            print(train_data.iloc[i,:].values)
            print(test_data.iloc[i,:].values)
            print([train_data.iloc[i,:].values,test_data.iloc[i,:].values])
            plot_data = pd.DataFrame()
            plot_data['train'] = train_data.iloc[i,:].values
            plot_data['test']  = test_data.iloc[i,:].values
            plot_data.index = range(epoch_num)
            '''
            plot_data = pd.DataFrame([train_data.iloc[i,:].values,test_data.iloc[i,:].values],
                                     index   = range(epoch_num),
                                     columns = ['train','test'])
            plot_data  = pd.DataFrame([train_data.iloc[i,:].values.T, test_data.iloc[i,:].values.T],
                                     index   = range(epoch_num),
                                     columns = ['train','test'])
            '''
            print(plot_data)
            fig = plt.figure()
            ax  = fig.add_subplot(1,1,1)
            plot_data.plot(ax=ax, 
                           xticks = range(0,epoch_num,15),
                           title = titles[i],
                           kind='line')
            fig.show()
            fig.savefig(os.path.join(args.output_dir, args.type+'_'+titles[i]+'.png'))

    elif args.type.endswith('dist'):
        config_names = configs.iloc[:,0].str.cat(configs.iloc[:,1:-1], sep='-', na_rep='*')
        print(config_names)
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
                                plot_file(filepath, args.output_dir, config_names, count, args.type)
                            elif args.data_types in ['a','w','e','g']:
                                if filename[11] == args.data_types: 
                                    #print('zk debug: find one', args.data_types)
                                    plot_file(filepath, args.output_dir, config_names, count, args.type)
                            else:
                                exit("wrong type")
                    count += 1

    elif args.type == 'curve':
        config_names = configs.iloc[:,0].str.cat(configs.iloc[:,1:-1], sep='-', na_rep='*')
        print(config_names)
    else:
        exit('wrong type arguement')



if __name__ == '__main__':

    main(sys.argv)

