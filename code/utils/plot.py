import os
import numpy as np
import sys
import matplotlib
matplotlib.use('Tkagg') # otherwise Linux server will crash using matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

__all__ = ['CurvePlot','LinePlot','DistriPlot']




def DistriPlot(curve_num, file_path, file_list=[], index_list=[], pic_name=None, mean=None, std=None):
    pass

# CurvePlot is to plot [max 75% 50% 25% min] curve of data varing with epoch
def CurvePlot(curve_num, file_path, file_list=[], index_list=[], pic_name=None, mean=None, std=None):

    count = 0
    file_name = os.path.join(file_path, file_list[0]+'.txt')
    with open(file_name) as f:                      # open txt file
        for line in f:                              # check line by line
            count += 1                              # count total line number
    name = []
    curve = np.zeros((curve_num,(count-2)))
    mean_curve = np.zeros(count-2)
    std_curve  = np.zeros(count-2)

    for i in range(curve_num):
        count = 0
        file_name = os.path.join(file_path, file_list[i]+'.txt')
        with open(file_name) as f:                      # open txt file
            for line in f:                              # check line by line
                datalist = line.split()                 # split one line to a list
                if count == 0:
                    name.append(datalist[index_list[i]]+'_'+file_list[i])
                    count += 1
                elif count == 1:
                    count += 1
                else:
                    curve[i][count-2] = float(datalist[index_list[i]])
                    count += 1

    # plot mean +- std only when ordered
    if (mean is not None) and (std is not None):
        count = 0
        file_name = os.path.join(file_path, mean+'.txt')
        with open(file_name) as f:                      # open mean file
            for line in f:                              # check line by line
                datalist = line.split()                 # split one line to a list
                if count == 0:
                    count += 1
                elif count == 1:
                    count += 1
                else:
                    mean_curve[count-2] = float(datalist[index_list[0]])
                    count += 1
        count = 0
        file_name = os.path.join(file_path, std+'.txt')
        with open(file_name) as f:                      # open std file
            for line in f:                              # check line by line
                datalist = line.split()                 # split one line to a list
                if count == 0:
                    count += 1
                elif count == 1:
                    count += 1
                else:
                    std_curve[count-2] = float(datalist[index_list[0]])
                    count += 1


    high_curve = mean_curve + std_curve
    low_curve  = mean_curve - std_curve

    #fig = plt.figure(figsize=(12,10))
    for i in range(curve_num):
        plt.plot(curve[i])
    if (mean is not None) and (std is not None):
        plt.plot(high_curve, color='silver')
        plt.plot(low_curve,  color='silver')
        plt.fill_between(np.arange(len(low_curve)),low_curve, high_curve, color='silver', alpha=0.25)

    #plt.legend(name)
    plt.legend(file_list)
    plt.grid(True)
    #plt.show()

    if pic_name is not None:
        #save_name = os.path.join(file_path, pic_name+'_curve0603.jpg')
        save_name = os.path.join(file_path, pic_name+'_curve0603.pdf')
        #pdf = PdfPages(save_name)
        plt.savefig(save_name)
        #pdf.savefig(save_name)
    plt.close()
    #pdf.close() 



# LinePlot can plot $line_num lines read from one file on one picture
def LinePlot(line_num, file_path, file_list=[], line_index=30, pic_name=None):

    name  = []
    lines = []
    legend= []
    for i in range(line_num):
        count = 0
        linefile = os.path.join(file_path, file_list[i]+'.txt')
        legend.append(file_list[i]+'_30')
        with open(linefile) as f:                       # open txt file
            for line in f:                              # check line by line
                datalist = line.split()                 # split one line to a list
                if count == 0 and i == 0:
                    name = datalist
                elif count == line_index:
                    lines.append([float(i) for i in datalist])
                else:
                    pass
                count += 1

    line = np.array(lines)
    fig = plt.figure(figsize=(14,7))
    for i in range(line_num):
        plt.plot(line[i])
    plt.legend(legend)
    plt.grid(True)
    #plt.show()

    if pic_name is not None:
        save_name = os.path.join(file_path, pic_name+'_line.jpg')
        plt.savefig(save_name)
    plt.close()



if __name__ == '__main__':

    # run to plot w a g r curve pictures of some specific layers
    # and cross layer line of w a g r of two specific epoch
    # and distribute of w a g of one specific epoch and layer

    file_path  = "/home/lijiajie16/fix_train_cnn_with_pytorch/checkpoint/resnet/test_hooker/log_20190522_23-35-54/"
    #file_path  = '../../checkpoint/float_baseline/cifar/resnet/log_test_debug/'

    # params of a curve plot
    file_list  = ['a_max', 'a_p75', 'a_p50', 'a_p25', 'a_min']
    curve_num  = len(file_list)
    mean = 'a_mean'
    std  = 'a_std'
    conv_index_list = [0,0,0,0,0]
    bn_index_list = [1,1,1,1,1]
    relu_index_list = [2,2,2,2,2]
    l02bn1_index_list = [16,16,16,16,16]
    l11bn2_index_list = [33,33,33,33,33]
    l20bnd_index_list = [42,42,42,42,42]
    l21bn1_index_list = [50,50,50,50,50]
    fc_index_list = [61,61,61,61,61,61]
    #CurvePlot(curve_num, file_path, file_list, conv_index_list, 'a_conv')
    CurvePlot(curve_num, file_path, file_list, conv_index_list,   'a_conv_vs_m' , mean, std)
    CurvePlot(curve_num, file_path, file_list,   bn_index_list,   'a_bn_vs_m'   , mean, std)
    CurvePlot(curve_num, file_path, file_list, relu_index_list,   'a_relu_vs_m' , mean, std)
    CurvePlot(curve_num, file_path, file_list, l02bn1_index_list, 'a_02bn1_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list, l11bn2_index_list, 'a_11bn2_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list, l20bnd_index_list, 'a_20bnd_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list, l21bn1_index_list, 'a_21bn1_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list,   fc_index_list,   'a_fc_vs_m'   , mean, std)

    # params of g curve plot
    file_list  = ['g_max', 'g_p75', 'g_p50', 'g_p25', 'g_min']
    curve_num  = len(file_list)
    mean = 'g_mean'
    std  = 'g_std'
    conv_index_list = [61,61,61,61,61,61]
    bn_index_list = [60,60,60,60,60,60]
    relu_index_list = [59,59,59,59,59,59]
    l02re1_index_list = [44,44,44,44,44]
    l11re2_index_list = [27,27,27,27,27]
    l20cod_index_list = [20,20,20,20,20]
    l21re1_index_list = [10,10,10,10,10]
    fc_index_list = [0,0,0,0,0]
    #CurvePlot(curve_num, file_path, file_list, conv_index_list, 'a_conv')
    CurvePlot(curve_num, file_path, file_list, conv_index_list,   'g_conv_vs_m' , mean, std)
    CurvePlot(curve_num, file_path, file_list,   bn_index_list,   'g_bn_vs_m'   , mean, std)
    CurvePlot(curve_num, file_path, file_list, relu_index_list,   'g_relu_vs_m' , mean, std)
    CurvePlot(curve_num, file_path, file_list, l02re1_index_list, 'g_02relu1_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list, l11re2_index_list, 'g_11relu2_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list, l20cod_index_list, 'g_20convd_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list, l21re1_index_list, 'g_21relu1_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list,   fc_index_list,   'g_fc_vs_m'   , mean, std)
    '''

    # params of w curve plot
    file_list  = ['w_max', 'w_p75', 'w_p50', 'w_p25', 'w_min']
    curve_num  = len(file_list)
    mean = 'w_mean'
    std  = 'w_std'
    #conv_index_list = [0,0,0,0,0]
    conv_index_list = [2,2,2,2,2]
    #bn_index_list = [1,1,1,1,1]
    #l02co1_index_list = [10,10,10,10,10]
    #l11bn2_index_list = [23,23,23,23,23]
    #l20cod_index_list = [28,28,28,28,28]
    #l21co2_index_list = [36,36,36,36,36]
    #fc_index_list = [42,42,42,42,42]
    #CurvePlot(curve_num, file_path, file_list, conv_index_list, 'a_conv')
    CurvePlot(curve_num, file_path, file_list, conv_index_list,   'w_conv_vs_m' , mean, std)
    CurvePlot(curve_num, file_path, file_list,   bn_index_list,   'w_bn_vs_m'   , mean, std)
    CurvePlot(curve_num, file_path, file_list, l02co1_index_list, 'w_02conv1_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list, l11bn2_index_list, 'w_11bn2_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list, l20cod_index_list, 'w_20convd_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list, l21co2_index_list, 'w_21conv2_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list,   fc_index_list,   'w_fc_vs_m'   , mean, std)

    # params of b curve plot
    file_list  = ['b_max', 'b_p75', 'b_p50', 'b_p25', 'b_min']
    curve_num  = len(file_list)
    mean = 'b_mean'
    std  = 'b_std'
    bn_index_list = [0,0,0,0,0]
    l02bn1_index_list = [5,5,5,5,5]
    l11bn2_index_list = [11,11,11,11,11]
    l20bnd_index_list = [14,14,14,14,14]
    fc_index_list = [21,21,21,21,21]
    #CurvePlot(curve_num, file_path, file_list, conv_index_list, 'a_conv')
    CurvePlot(curve_num, file_path, file_list,   bn_index_list,   'b_bn_vs_m'   , mean, std)
    CurvePlot(curve_num, file_path, file_list, l02bn1_index_list, 'b_02bn1_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list, l11bn2_index_list, 'b_11bn2_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list, l20bnd_index_list, 'b_20bnd_vs_m', mean, std)
    CurvePlot(curve_num, file_path, file_list,   fc_index_list,   'b_fc_vs_m'   , mean, std)

    # params of r curve plot
    file_list  = ['sparse_ratio','sparse_ratio','sparse_ratio','sparse_ratio','sparse_ratio']
    curve_num  = len(file_list)
    layers_list = [1,3,5,6,7]
    #CurvePlot(curve_num, file_path, file_list, conv_index_list, 'a_conv')
    CurvePlot(curve_num, file_path, file_list, layers_list, 'density_of_each_l')

    # params of w a g line plot
    file_list  = ['a_max', 'a_p75', 'a_p50', 'a_p25', 'a_min']
    line_num  = len(file_list)
    line_index = 30
    LinePlot(line_num, file_path, file_list, line_index, 'cross_layer_a30')

    file_list  = ['g_max', 'g_p75', 'g_p50', 'g_p25', 'g_min']
    line_num  = len(file_list)
    line_index = 30
    LinePlot(line_num, file_path, file_list, line_index, 'cross_layer_g30')

    file_list  = ['w_max', 'w_p75', 'w_p50', 'w_p25', 'w_min']
    line_num  = len(file_list)
    line_index = 30
    LinePlot(line_num, file_path, file_list, line_index, 'cross_layer_w30')

    file_list  = ['sparse_ratio']
    line_num  = len(file_list)
    line_index = 30
    LinePlot(line_num, file_path, file_list, line_index, 'cross_layer_density_30')
    '''

