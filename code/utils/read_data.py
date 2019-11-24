import os
import numpy as np
import sys

count = 0
features = np.zeros((3*224*224))
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
