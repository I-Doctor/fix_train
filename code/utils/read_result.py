######################################################################
# (c) Copyright EFC of NICS, Tsinghua University. All rights reserved.
#  Author: Kai Zhong
#  Email : zhongk15@mails.tsinghua.edu.cn
#  
#  Create Date : 2019.03.16
#  File Name   : read_result.py
#  Description : read the config of train and test accuracy data from
#                log file and show on one screen to compare
#  Dependencies: 
######################################################################

import os
import numpy as np
import sys

def main(argv):
    
    parser = argparse.ArgumentParser()

    # Required arguments: input and output files.
    parser.add_argument(
        "value_name",
        help = "What are you going to read from log files.\n        \
                supported: best_acc test_acc train_acc "
    )
    parser.add_argument(
        "output_file",
        help = "The name of output file to store the result."
    )

    count = 0
    print "features shape: ", features.shape
    with open("./fix_results/fix_test.log") as f:   # open txt file
        for line in f:                                          # check line by line
            datalist = line.split()                             # split one line to a list
            if len(datalist) > 8:                               # jump over void line
                if datalist[3] == 'net_test.cpp:305]' and datalist[4] == 'Batch' and datalist[6] == 'data':
                    features[count] = np.float32( datalist[8] )
                    count += 1
    features = features.reshape(3,224,224)
    print "features shape: ", features.shape
    print "features fixed: ", features
