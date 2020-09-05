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



def check_column(configs, column_label):
    ''' check if there is already column named column_label '''

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
            configs.loc[count,(column_label)] = wordlist[2] \
                    if column_label != 'output_dir' else wordlist[2][-17:]
        else:
            configs[column_label] = None
            configs.loc[count,(column_label)] = wordlist[2] \
                    if column_label != 'output_dir' else wordlist[2][-17:]

    # second level configs
    elif pos == 3:
        # deal with q_cfg
        if wordlist[2] == 'q_cfg':
            for i in range(4, len(wordlist)):
                if wordlist[i].endswith("':"):
                    column_label = wordlist[i]
                    data_element = wordlist[i+1]
                    for j in range(i+2, len(wordlist)):
                        if wordlist[j].endswith("':"): break
                        else: data_element += wordlist[j]
                    if check_column(configs, column_label):
                        configs.loc[count,(column_label)] = data_element
                    else:
                        configs[column_label] = None
                        configs.loc[count,(column_label)] = data_element
        # len > 5 means list configs
        elif len(wordlist) > 5:
            column_label = wordlist[0]+wordlist[2]
            data_element = wordlist[4] 
            for i in range(5, len(wordlist)):
                data_element += wordlist[i]
            if check_column(configs, column_label):
                configs.loc[count,(column_label)] = data_element
            else:
                configs[column_label] = None
                configs.loc[count,(column_label)] = data_element
        # !len > 5 means one element configs
        else:
            column_label = wordlist[0]+wordlist[2]
            if check_column(configs, column_label):
                configs.loc[count,(column_label)] = wordlist[4]
            else:
                configs[column_label] = None
                configs.loc[count,(column_label)] = wordlist[4]
    else:
        print(wordlist, pos)
        exit("wrong : position")



def add_results(results, count, column_label, column_data):
    ''' add one result into results
    '''
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
            if len(wordlist) >= 5 and wordlist[0] != 'accuracy'\
            and wordlist[0] != 'log':
                if wordlist[3]==':':
                    add_line(configs, count, wordlist, 3)         # add this line to configs
            # process long config lines with : at position 1
            elif len(wordlist) >= 3 and wordlist[0] != 'gpu':
                if wordlist[1]==':': 
                    add_line(configs, count, wordlist, 1)         # add this line to configs

            # process best result
            if len(wordlist) > 1:
                # add best acc
                if wordlist[0] == 'best':
                    add_results(results, count, 'bestacc', wordlist[2])
                    add_results(results, count, 'bestepoch', wordlist[5])
                # add train loss and acc
                elif wordlist[0] == 'epoch:':
                    train_acc  = wordlist[13][1:-1]
                    train_loss = wordlist[10][1:-1]
                # add test loss
                elif wordlist[0] == 'test:':
                    test_loss = wordlist[7][1:-1]
                # add test acc and save all results in this epoch to results
                elif wordlist[0] == '*':
                    add_results(results, count, str(temp_epoch)+'trainacc', train_acc)
                    add_results(results, count, str(temp_epoch)+'trainloss', train_loss)
                    add_results(results, count, str(temp_epoch)+'testloss', test_loss)
                    add_results(results, count, str(temp_epoch)+'testacc', wordlist[2])
                    add_results(results, count, str(temp_epoch)+'test5acc', wordlist[4])
                    temp_epoch += 1

        return temp_epoch
                


def main(argv):

    print(argparse)
    print(type(argparse))

    parser = argparse.argumentparser()

    # required arguments: 
    parser.add_argument(
        "type",
        help = "what type of mission are you going to do.\n\
                supported: compare loss_curve acc_curve data_range"
    )
    parser.add_argument(
        "output_dir",
        help = "the name of output dir to store the results."
    )
    parser.add_argument(
        "--results_name",
        help = "what results are you going to plot or compare.\n        \
                supported: best_acc test_acc train_acc test_loss train_loss"
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
    print(args.file_range)

    dirlist = os.listdir('./')
    print(dirlist)

    configs = pd.dataframe()
    print(configs)
    results = pd.dataframe()
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
    #print(configs)
    trainloss = results[['%dtrainloss'%i for i in range(epoch_num+1)]]
    trainacc  = results[['%dtrainacc'%i for i in range(epoch_num+1)]]
    testloss = results[['%dtestloss'%i for i in range(epoch_num+1)]]
    testacc  = results[['%dtestacc'%i for i in range(epoch_num+1)]]
    
    values = {'train_loss': trainloss, 'train_acc': trainacc,
              'test_loss': testloss, 'test_acc': testacc}

    # output datas with args
    if args.type == 'compare':
        # compare best acc throught table
        if args.results == 'best_acc':
            output = pd.concat(configs, results[['bestacc','bestepoch']])
            print(output)
            output.to_csv(os.path.join(args.output_dir,'compare_best.csv'))
        # compare one curve throught figure
        elif args.results.endswith('acc') or args.results.endswiht('loss'):
            values[args.results_name].plot(kind='line',marker='o')
        # compare all curves throught figures
        else:
            exit("can't compare all results of different configs")
    # plot one curve of each config onto oue figure
    elif args.type == 'loss_curve':
        pass
    elif args.type == 'acc_curve':
        pass
    elif args.type == 'data_range':
        pass

    return



if __name__ == '__main__':

    main(sys.argv)

