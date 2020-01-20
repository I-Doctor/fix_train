
# import basic libs
import os,sys
import six
import math
import time
import shutil
import random
import datetime
import warnings
import argparse
import numpy as np  
import matplotlib.pyplot as plt 
import hiddenlayer as hl
from collections import OrderedDict

# import pytorch libs
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# import customized libs
from model import *
from utils import *

class MyDataParallel(nn.DataParallel):     
    def __getattr__(self, name):         
        print(self)
        print(type(self))
        m=getattr(self,'module')
        print(m)
        raise ValueError('data parrallel')
        
        return getattr(module,name)


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and
                     callable(models.__dict__[name]))

best_acc1 = 0
best_epo1 = 0
parallel = False

def main(argv):

    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    # Required arguments: input and output files.
    parser.add_argument(
        "data_root", 
        help = "Must give the dataset root to do the training (absolute path)."
   )
    parser.add_argument(
        "config_file",
        help = "Must give the config file of this experiment."
   )
    parser.add_argument(
        "output_dir",
        help = "Must give the path to save output logfile and other data."
   )
    parser.add_argument(
        "--gpu", default=None, type=int, 
        help = "Set visible gpu independantly because it has no influence to results"
   )
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                         help='Use multi-processing distributed training to launch '
                              'N processes per node, which has N GPUs. This is the '
                              'fastest way to use PyTorch for either single node or '
                              'multi node data parallel training'
   )
    #parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
    #                     choices=model_names,
    #                     help='model architecture: '
    #                     + ' | '.join(model_names)
    #                     + ' (default: resnet18)')
    #parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
    #                     help='number of data loading workers (default: 4)')
    #parser.add_argument('-p', '--print-freq', default=10, type=int,
    #                                        metavar='N', help='print frequency (default: 10)')
    #parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                                        help='path to latest checkpoint (default: none)')
    #parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
    #                                        help='evaluate model on validation set')
    #parser.add_argument('--pretrained', dest='pretrained', action='store_true',
    #                                        help='use pre-trained model')
    #parser.add_argument('--world-size', default=-1, type=int,
    #                                        help='number of nodes for distributed training')
    #parser.add_argument('--rank', default=-1, type=int,
    #                                        help='node rank for distributed training')
    #parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
    #                                        help='url used to set up distributed training')
    #parser.add_argument('--dist-backend', default='nccl', type=str,
    #                                        help='distributed backend')
    #parser.add_argument('--seed', default=None, type=int,
    #                                        help='seed for initializing training. ')
    #parser.add_argument('--gpu', default=None, type=int,
    #                                        help='GPU id to use.')
    #parser.add_argument('--multiprocessing-distributed', action='store_true',
    #                     help='Use multi-processing distributed training to launch '
    #                          'N processes per node, which has N GPUs. This is the '
    #                          'fastest way to use PyTorch for either single node or '
    #                          'multi node data parallel training')


    #-----------------------LOAD AND CHECK PARAMETER---------------------#
    args = parser.parse_args()

    print("cfg_file name:", args.config_file)
    cfg = create_default_cfg()  
    update_cfg(cfg, args.config_file)  

    net_cfg   = cfg.NETWORK
    train_cfg = cfg.TRAIN
    data_cfg  = cfg.DATA
    log_cfg   = cfg.LOG

    cfg.data_root = args.data_root
    cfg.output_dir = args.output_dir
    cfg.gpu = args.gpu
    cfg.multiprocessing_distributed = args.multiprocessing_distributed

    print_cfg(cfg)
    
    # check learning rate adjustment parameters are valid
    if isinstance(train_cfg.learning_rate, list) & (isinstance(train_cfg.decay_step, list)):
        if len(train_cfg.learning_rate) == len(train_cfg.decay_step):
            if len(train_cfg.learning_rate) == 1:
                warnings.warn('You are using only one learning rate during the training')
            else:
                print('Using learning rate adjustment')
        else:
            raise ValueError('The length of learning rate and decay step are not equal')
    else:
       raise ValueError('The learning rate or decay step is not list')

    # check random seed and record
    if cfg.random_seed is not None:
        random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    # check and set whole system
    if args.gpu is not None:
        #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        warnings.warn('You have chosen a specific GPU. This will completely '
                       'disable data parallelism.')
    
    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])
    
    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed
    
    ngpus_per_node = torch.cuda.device_count()
    
    if cfg.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.world_size = ngpus_per_node * cfg.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    global best_acc1
    global best_epo1
    global parallel
    cfg.gpu = gpu

    net_cfg = cfg.NETWORK
    train_cfg = cfg.TRAIN
    data_cfg = cfg.DATA
    log_cfg = cfg.LOG

    # record learning rate parameters 
    LR_start = train_cfg.learning_rate[0]
    START_epoch = 0

    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))
    
    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)


    # create model
    if net_cfg.pretrained:
        print("=> using pre-trained model '{}-{}'".format(net_cfg.arch, net_cfg.depth))
        #model = models.__dict__[net_cfg.arch+str(net_cfg.depth)](pretrained=True)
        model = resnet(net_cfg.depth, net_cfg.num_classes, net_cfg.q_cfg)
    else:
        print("=> creating model '{}-{}'".format(net_cfg.arch, net_cfg.depth))
        #model = models.__dict__[net_cfg.arch+str(net_cfg.depth)]()
        model = resnet(net_cfg.depth, net_cfg.num_classes, net_cfg.q_cfg)

    
    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            data_cfg.batch_size = int(data_cfg.batch_size / ngpus_per_node)
            data_cfg.num_works = int(data_cfg.num_works / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if net_cfg.arch.startswith('alexnet') or net_cfg.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            #model = MyDataParallel(model).cuda()
            parallel = True
            model = torch.nn.DataParallel(model).cuda()
    

    if data_cfg.dataset=='ILSVRC2012_img':
        df = torch_summarize_df(input_size=(3,256,256), model=model)
    else:
        df = torch_summarize_df(input_size=(3,32,32), model=model)
    print(df)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    print("===start defining optimizer===")
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(cfg.gpu)
    
    opt_Adam  = optim.Adam( model.parameters(), lr = LR_start, 
                            betas=(0.9, 0.999), weight_decay = 1e-8) 
    opt_SGD   = optim.SGD ( model.parameters(), lr = LR_start,
                            weight_decay=train_cfg.weight_decay) 
    opt_SGDm  = optim.SGD ( model.parameters(), lr = LR_start, 
                            momentum=train_cfg.momentum, 
                            weight_decay=train_cfg.weight_decay) 
    opt_RMS   = optim.RMSprop(model.parameters(), lr = LR_start, 
                              weight_decay=5e-4)
    if train_cfg.optimizer == "Adam":
        print("  Use optimizer Adam.")
        optimizer = opt_Adam    
    elif train_cfg.optimizer == "SGDm":
        print("  Use optimizer SGDm.")
        optimizer = opt_SGDm
    else:
        print("  Use optimizer SGD.")
        optimizer = opt_SGD
    
    
    # optionally resume from a checkpoint
    if train_cfg.resume is not None:
        if os.path.isfile(train_cfg.resume):
            print("=> loading checkpoint '{}'".format(train_cfg.resume))
            checkpoint = torch.load(train_cfg.resume)
            START_epoch = checkpoint['epoch']
            parallel = checkpoint['parallel']
            best_acc1 = checkpoint['best_acc1']
            best_epo1 = checkpoint['best_epo1']
            print("=> checkpoint best '{}' @ {}".format(best_acc1,best_epo1))
            if cfg.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(cfg.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            #print('debug:',checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                   .format(train_cfg.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(train_cfg.resume))
        if parallel:
            model.module.enable_quantize()
        else:
            model.enable_quantize()
    
    cudnn.benchmark = True
    #print(model)
    #print(torch_summarize(model))


    print("===start loading data===")
    # Data loading code
    traindir = os.path.join(cfg.data_root, data_cfg.dataset+'_train')
    valdir = os.path.join(cfg.data_root, data_cfg.dataset+'_val')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=data_cfg.pixel_means,
                                     std=data_cfg.pixel_stds)
    
    if data_cfg.dataset=='ILSVRC2012_img':
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(
            valdir, 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    else: # cifar 10
        train_dataset = datasets.CIFAR10(
            cfg.data_root+'cifar10',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.CIFAR10(
            cfg.data_root+'cifar10',
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    
    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if_shuffle=(train_sampler is None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=data_cfg.batch_size, shuffle=if_shuffle,
        num_workers=data_cfg.num_works, pin_memory=True, sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=data_cfg.batch_size, shuffle=False,
        num_workers=data_cfg.num_works, pin_memory=True)
    
    if train_cfg.evaluate:
        validate(val_loader, model, criterion, cfg)
        if data_cfg.dataset=='ILSVRC2012_img':
            pass
        else:
            finalreport(val_loader, model, net_cfg.num_classes, cfg.cuda)
        return
    
    if log_cfg.frequent < 200:
        print("===creat hook_ctrls===")
        a_hooks = Hook_ctrl(model, 'a')
        #w_hooks = Hook_ctrl(model, 'w')
        #g_hooks = Hook_ctrl(model, 'g')
        #e_hooks = Hook_ctrl(model, 'e')
        #hooks = [a_hooks,w_hooks,e_hooks,g_hooks]
        hooks = [a_hooks]
    print("===start train with epoch===")
    for epoch in range(START_epoch, train_cfg.epoch):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, train_cfg)
        if epoch == cfg.float_epoch:
            if parallel:
                model.module.enable_quantize()
            else:
                model.enable_quantize()
        
        # train for one epoch
        if epoch % log_cfg.frequent == 0 and log_cfg.frequent<200 :
            a_hooks.hook_insert()
            #w_hooks.hook_insert()
            #g_hooks.hook_insert()
            #e_hooks.hook_insert()
            train(train_loader, model, criterion, optimizer, epoch, cfg, hooks)
            a_checkpoint = os.path.join(cfg.output_dir, 'checkpoint_a_%s.h5'%(epoch))
            #w_checkpoint = os.path.join(cfg.output_dir, 'checkpoint_w_%s.h5'%(epoch))
            #g_checkpoint = os.path.join(cfg.output_dir, 'checkpoint_g_%s.h5'%(epoch))
            #e_checkpoint = os.path.join(cfg.output_dir, 'checkpoint_e_%s.h5'%(epoch))
            a_hooks.save(ctype=[], output_path = a_checkpoint, resume = False)
            #w_hooks.save(ctype=[], output_path = w_checkpoint, resume = False)
            #g_hooks.save(ctype=[], output_path = g_checkpoint, resume = False)
            #e_hooks.save(ctype=[], output_path = e_checkpoint, resume = False)
        else:
            train(train_loader, model, criterion, optimizer, epoch, cfg)
        
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, cfg)
        
        # remember best_acc1 @ best_epo1 and save checkpoint
        if acc1 > best_acc1 :
            best_acc1 = acc1
            best_epo1 = epoch 
            if not cfg.multiprocessing_distributed or \
            (cfg.multiprocessing_distributed and cfg.rank % ngpus_per_node == 0):
                print(" --- save checkpoint  for best ---")
                checkpoint = {
                    'epoch': epoch + 1,
                    'parallel': parallel,
                    'arch': net_cfg.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc1': best_acc1,
                    'best_epo1': best_epo1,
                }
                torch.save(checkpoint, os.path.join(cfg.output_dir, 'checkpoint_best_%s.pkl'%(epoch//20)))

        # save checkpoint frequentily
        if epoch % cfg.checkpoint_freq == 0:
            if not cfg.multiprocessing_distributed or \
            (cfg.multiprocessing_distributed and cfg.rank % ngpus_per_node == 0):
                print(" --- save checkpoint  frequently ---")
                checkpoint = {
                    'epoch': epoch + 1,
                    'parallel': parallel,
                    'arch': net_cfg.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc1': best_acc1,
                    'best_epo1': best_epo1,
                }
                torch.save(checkpoint, os.path.join(cfg.output_dir, 'checkpoint_%s.pkl'%epoch))

    if data_cfg.dataset=='ILSVRC2012_img':
        acc1 = validate(val_loader, model, criterion, cfg, True)
        acc1 = validate(val_loader, model, criterion, cfg, True)
        acc1 = validate(val_loader, model, criterion, cfg, True)
        acc1 = validate(val_loader, model, criterion, cfg)
        acc1 = validate(val_loader, model, criterion, cfg)
        pass
    else:
        finalreport(val_loader, model, net_cfg.num_classes, cfg.cuda)
    print('BEST RESULT: %.3f%% at EPOCH: %d' % (best_acc1, best_epo1))


def train(train_loader, model, criterion, optimizer, epoch, args, hooks=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to train mode
    model.train()
    
    print("---train(epoch)---")
    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):

        if i == 1 and hooks is not None:
            print("--remove hooks--")
            for h in hooks:
                h.remove()

        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        # compute output
        output = model(inputs)
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            #return


def validate(val_loader, model, criterion, args, train=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    model.train(train)
    
    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            # compute output
            output = model(inputs)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
        
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))
    
    return top1.avg

'''
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
'''


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate every time coming to specific epochs"""
    for i in range (1, len(args.decay_step)):
        if epoch == args.decay_step[i]:
            lr_now = args.learning_rate[i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_now


def finalreport(testloader, net, num_class, use_cuda):

    """ Computes the final report of the net on testset """

    net.eval()

    classes = []
    if num_class == 10:
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 
    class_correct = list(0. for i in range(num_class))  
    class_total = list(0. for i in range(num_class))

    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            if use_cuda and torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            outputs = net(images)      # predicted shape -> (4, 1)  
            _, predicted = torch.max(outputs.data, 1)  
            c = (predicted == labels)  # squeeze(), return a tensor with all the dimensions of input of size 1 removed  
            c = c.squeeze()            # translate shape from (4*1) to (4)  
            
            for i in range(4):  
                label = labels[i]  
                class_correct[label] += c[i]  
                class_total[label] += 1  
          
    for i in range(num_class):  
        print('Accuracy of %5s : %.4f' 
            % ((classes[i] if(num_class==10) else i), (float(class_correct[i])/class_total[i])))

    print('Total accuracy: %.4f' % (sum(class_correct)/sum(class_total)))

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    main(sys.argv)
