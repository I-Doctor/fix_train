import numpy as np
import os
import json
import shutil

__all__ = ['Logger']

class Logger(object):

    r"""
        Parameters
        ----------
        path: string
            path to directory to save data files
        title: string
            title of log file WITHOUT extension
        params: data names and number per name
            will be saved in the first two lines in file
        resume: bool
            resume previous log file
        data_format: str('csv'|'json'|'txt')
            which file format to use to save the data
    """
    supported_data_formats = ['csv', 'json', 'txt']

    def __init__(self, path='', title='', params=None, resume=False, data_format='txt'):

        if data_format not in Logger.supported_data_formats:
            raise ValueError('data_format must be csv or json or txt')

        if data_format == 'json':
            self.data_path = os.path.join(path,'{}.json'.format(title))
        elif data_format == 'csv':
            self.data_path = os.path.join(path,'{}.csv'.format(title))
        else:
            self.data_path = os.path.join(path,'{}.txt'.format(title))

        self.file=None
        self.names = []
        self.num_per_name = params
        self.length = 0

        if resume:
            if not os.path.isfile(self.data_path):
                raise ValueError('path/title.data_fromat is not a exist file to resume')
            else:
                self.resume()
        else:
            self.init()


    def resume(self):

        self.file = open(self.data_path, 'r') 
        names = self.file.readline()
        if names == None:
            self.file.close()
            pass

        numbers = self.file.readline()
        self.names = names.rstrip().split('\t')
        numbers = numbers.rstrip().split('\t')
        self.num_per_name = {}
        for i, name in enumerate(self.names):
            self.num_per_name[name] = (int)(numbers[i])

        self.file.close()


    def init(self):

        self.file = open(self.data_path, 'w') 

        if self.num_per_name == None:
            self.file.close()
            pass

        for key,val in self.num_per_name.items():
            self.file.write(key)
            self.file.write('\t')
            self.names.append(key)
        self.file.write('\n')
        for key,val in self.num_per_name.items():
            self.file.write("{:<2d}".format(val))
            self.file.write('\t')
            self.length += val
        self.file.write('\n')
        self.file.flush()

        self.file.close()


    def save(self, datas=None):

        if isinstance(datas,list):
            datas = np.array(datas)
        datas = datas.reshape(-1)
        if (len(datas)!=self.length)and(self.length>0): 
            raise ValueError("data length mismatch expect {}".format(self.length)) 

        self.file = open(self.data_path, 'a')

        for data in datas:
            self.file.write("{0:.12f}".format(float(data)))
            self.file.write('\t')
        self.file.write('\n')
        self.file.flush()

        self.file.close()
        

    def print(self, string=None):

        self.file = open(self.data_path, 'a')
        print(string, file=self.file) 
        self.file.close()


