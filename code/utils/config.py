import yaml
import random
import numpy as np
from easydict import EasyDict as edict

__all__ = ['create_default_cfg', 'update_cfg', 'print_cfg']

def create_default_cfg():

    config = edict()

    # Must give the dataset root to do the training (absolute path)."
    config.data_root = ''
    # Must give the path to save output logfile and other data."
    #config.output_dir = ''
    ''' output dir can not define with yaml because it is given every time you run the code'''
    # The manual random seed of training."
    config.random_seed = None
    # Using gpu computation."
    config.cuda = True
    # Using parallel training."
    config.parallel = True
    # Using multiprocessing training."
    config.multiprocessing_distributed = True
    # Using disgtributed training."
    config.distributed = False
    # Parameters for distributed training."
    config.world_size = -1
    config.rank = -1
    config.dist_url = 'tcp://127.0.0.1:9000'
    config.dist_backend = 'nccl'
    # The number of available gpu device."
    config.visible_device = "0"
    ''' cuda visible device should be given every time you run the code'''
    # The epoch frequence of saving snapshot."
    config.checkpoint_freq = 30
    # The batch frequence of output."
    config.print_freq = 30
    # The epoch starting quantize."
    config.float_epoch = 0
    
    config.NETWORK = edict()
    # Depth of ResNet model."
    config.NETWORK.arch = 'resnet'
    config.NETWORK.depth = 20
    config.NETWORK.num_classes = 10
    # using batch normalization when traning."
    config.NETWORK.batchnorm = True
    config.NETWORK.pretrained = False
    # using quantization when traning."
    config.NETWORK.quantize = False
    config.NETWORK.q_cfg = edict()
    # configs
    config.NETWORK.q_cfg.bw = [8,8,8]
    config.NETWORK.q_cfg.level = [0,0,0]
    config.NETWORK.q_cfg.signed = False
    config.NETWORK.q_cfg.stochastic = True
    config.NETWORK.q_cfg.qfirst = False
    config.NETWORK.q_cfg.qlinear = False
    config.NETWORK.q_cfg.qbn = False
    config.NETWORK.q_cfg.erange = ['max','max','max']
    config.NETWORK.q_cfg.linear = [None, None, None]
    config.NETWORK.q_cfg.hard = 'pow'
    config.NETWORK.q_cfg.group = [False, False, False]
    config.NETWORK.q_cfg.g_scale_type = 'simple'
    config.NETWORK.q_cfg.value_type = 'progess'

    config.TRAIN = edict()
    # Resume or not"
    config.TRAIN.resume = None
    # Evaluate or not"
    config.TRAIN.evaluate = False
    # The number of epoch of training."
    config.TRAIN.epoch = 165
    # The optimizer of training.(SGD,Adam,SGDm)"
    config.TRAIN.optimizer = 'SGDm'
    # Learning rate (list if change during the training)."
    config.TRAIN.learning_rate = [0.1, 0.001, 0.0001]
    # The corresponding learning rate will assign at decay step (list)."
    config.TRAIN.decay_step = [0, 81, 122]
    # using warmup at the beginning of training(just for log, lr&step can set above)."
    config.TRAIN.warmup = False
    # Decay factor of weight."
    config.TRAIN.weight_decay = 5e-4
    # Momentum using in optimizers."
    config.TRAIN.momentum = 0.9

    config.DATA = edict()
    # The size of mini-batch of SGD."
    config.DATA.dataset = 'cifar10'
    config.DATA.batch_size = 128
    config.DATA.num_works = 8
    # whether flip image
    config.DATA.flip = True
    # whether shuffle image
    config.DATA.shuffle = True
    # random crop image pad (no crop if 0)
    config.DATA.random_crop_pad = 4
    config.DATA.pixel_means = np.array([0.4914, 0.4822, 0.4465])
    config.DATA.pixel_stds  = np.array([0.2023, 0.1994, 0.2010])
    config.DATA.valid_batch_size = 128
    config.DATA.valid_num_works = 8

    config.LOG = edict()
    config.LOG.frequent = 200
    config.LOG.types = None

    return config


def update_cfg(config, config_file):

    exp_config = None

    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                # k is 'NETWORK', v is dict
                if isinstance(v, dict):
                    if k == 'TRAIN':
                        if 'learning_rate' in v:
                            v['learning_rate'] = tuple(v['learning_rate'])
                        if 'decay_step' in v:
                            v['decay_step'] = tuple(v['decay_step'])
                    elif k == 'DATA':
                        if 'pixel_vars' in v:
                            v['pixel_vars'] = np.array(v['pixel_vars'])
                        elif 'pixel_means' in v:
                            v['pixel_means'] = np.array(v['pixel_means'])
                    for vk, vv in v.items():
                        # k.vk is 'NETWORK.q_cfg', vv is dict
                        if isinstance(vv, dict):
                            # k.vk.vvk is 'NETWORK.q_cfg.bw'
                            for vvk, vvv in vv.items():
                                config[k][vk][vvk] = vvv
                        # vk is 'NETWORK.q_cfg', vv is dict
                        else:
                            config[k][vk] = vv
                # k is 'float_epoch'
                else:
                    config[k] = v
            else:
                raise ValueError("key must exist in config.py")


def print_cfg(config):

    print('##[config]##:')
    for k, v in config.items():
        if isinstance(v, dict):
            for vk, vv in v.items():
                print(k, '.', vk, ': ', vv)
            print(' ')
        else:
            print(k, ': ', v)


if __name__ == '__main__':

    cfg = create_default_cfg()
    print(cfg)
    with open('../../config/test_config_py/default.yaml', 'w') as outyaml:
        yaml.dump(cfg, outyaml, default_flow_style=False)
    #update_cfg(cfg, '../../config/test_config_py/test.yaml')
    update_cfg(cfg, '../../config/cifar10/fix_cfg/cifar-check.yaml')
    print(cfg.NETWORK.q_cfg.qlinear)
    print(cfg.NETWORK.q_cfg.linear)
    qcfg = cfg.NETWORK.q_cfg
    print(qcfg["linear"][0])
    print(cfg['TRAIN']['learning_rate'])
    print_cfg(cfg)
    print(cfg)


