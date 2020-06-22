# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 10:52:26 2018

@author: carri
"""
#区别于deeplab_co_attention_concat在于采用了新的model（siamese_model_concat_new）来train

import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
from utils.balanced_BCE import class_balanced_cross_entropy_loss
import os.path as osp
#from psp.model import PSPNet
#from dataloaders import davis_2016 as db
from dataloaders import TripletImg_video1 as db #采用voc dataset的数据设置格式方法
import matplotlib.pyplot as plt
import random
import timeit
import time
#from psp.model1 import CoattentionNet  #基于pspnet搭建的co-attention 模型
from deeplab.siamese_model_conf_gnn3_sa_org import CoattentionNet #siame_model 是直接将attend的model之后的结果输出
#from deeplab.utils import get_1x_lr_params, get_10x_lr_params#, adjust_learning_rate #, loss_calc
start = timeit.default_timer()

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PSPnet Network")

    # optimatization configuration
    parser.add_argument("--is-training", action="store_true", 
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default= 0.00025, 
                        help="Base learning rate for training with polynomial decay.") #0.001
    parser.add_argument("--weight-decay", type=float, default= 0.0005, 
                        help="Regularization parameter for L2-loss.")  # 0.0005
    parser.add_argument("--momentum", type=float, default= 0.9, 
                        help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default= 0.9, 
                        help="Decay parameter to compute the learning rate.")
    # dataset information
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        help="voc12, cityscapes, or pascal-context.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")

    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--random-seed", type=int, default= 1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument('--logFile', default='log.txt', 
                        help='File that stores the training and validation logs')
    # GPU configuration
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="3", help="choose gpu device.") #使用3号GPU


    return parser.parse_args()

args = get_arguments()


def configure_dataset_init_model(args):

    args.batch_size = 16# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
    args.maxEpoches = 60 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
    args.data_dir = '/home/ubuntu/xiankai/dataset/DAVIS-2016'   # 37572 image pairs
    args.img_dir = '/raid/dataset'
    args.data_list = './dataset/list/VOC2012/train_aug.txt'  # Path to the file listing the images in the dataset
    args.ignore_label = 255     #The index of the label to ignore during the training
    args.input_size = '378, 378' #Comma-separated string with height and width of images
    args.num_classes = 2      #Number of classes to predict (including background)
    args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
    #Where restore model pretrained on other dataset, such as COCO.")
    args.restore_from = './pretrained/deep_labv3/deeplab_davis_12_0.pth'#'./pretrained/deep_labv3/deeplab_davis_12_0.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
    args.snapshot_dir = './snapshots/davis_iteration_conf_gnn3_sa_org_init/'          #Where to save snapshots of the model
    args.resume = './snapshots/davis_iteration_conf_gnn3/co_attention_davis_149.pth' #checkpoint log file, helping recovering training


def adjust_learning_rate(optimizer, i_iter, epoch, max_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    
    lr = lr_poly(args.learning_rate, i_iter, max_iter, args.power, epoch)
    #optimizer.param_groups[0]['lr'] = lr
    if i_iter%3 ==0:
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = 0
    else:
        optimizer.param_groups[0]['lr'] = 0.01*lr
        optimizer.param_groups[1]['lr'] = lr * 10
        
    return lr

def loss_calc1(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    labels = torch.ge(label, 0.5).float()
#    
    batch_size = label.size()
    #print(batch_size)
    num_labels_pos = torch.sum(labels) 
#    
    batch_1 =  batch_size[0]* batch_size[2]
    batch_1 = batch_1* batch_size[3]
    weight_1 = torch.div(num_labels_pos, batch_1) # pos ratio
    weight_1 = torch.reciprocal(weight_1)
    #print(num_labels_pos, batch_1)
    weight_2 = torch.div(batch_1-num_labels_pos, batch_1)
    #print('postive ratio', weight_2, weight_1)
    weight_22 = torch.mul(weight_1,  torch.ones(batch_size[0], batch_size[1], batch_size[2], batch_size[3]).cuda())
    #weight_11 = torch.mul(weight_1,  torch.ones(batch_size[0], batch_size[1], batch_size[2]).cuda())
    criterion = torch.nn.BCELoss(weight = weight_22)#weight = torch.Tensor([0,1]) .cuda() #torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    #loss = class_balanced_cross_entropy_loss(pred, label).cuda()
        
    return criterion(pred, label)

def loss_calc2(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # Variable(label.long()).cuda()
    criterion = torch.nn.L1Loss()#.cuda() #torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    
    return criterion(pred, label)



def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []
    if torch.cuda.device_count() == 1:
        #b.append(model.encoder.conv1)
        #b.append(model.encoder.bn1)
        #b.append(model.encoder.layer1)
        #b.append(model.encoder.layer2)
        #b.append(model.encoder.layer3)
        #b.append(model.encoder.layer4)
        b.append(model.encoder.layer5)
    else:
        b.append(model.module.encoder.conv1)
        b.append(model.module.encoder.bn1)
        b.append(model.module.encoder.layer1)
        b.append(model.module.encoder.layer2)
        b.append(model.module.encoder.layer3)
        b.append(model.module.encoder.layer4)
        b.append(model.module.encoder.layer5)
        b.append(model.module.encoder.main_classifier)
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b_module = []
    if torch.cuda.device_count() == 1:
        b.append(model.linear_e.parameters())
        b.append(model.main_classifier.parameters())
    else:

        b.append(model.module.linear_e.parameters())
        b.append(model.module.ConvGRU.parameters())
        b.append(model.module.conv1.parameters())
        b.append(model.module.gate.parameters())
        b.append(model.module.bn1.parameters())
        b.append(model.module.main_classifier1.parameters())
        b.append(model.module.conv_fusion.parameters())

        
    for j in range(len(b)):
        for i in b[j]:
            yield i
            
def lr_poly(base_lr, iter, max_iter, power, epoch):
    if epoch<=2:
        factor = 1
    elif epoch>2 and epoch< 6:
        factor = 1
    else:
        factor = 1
    return base_lr*factor*((1-float(iter)/max_iter)**(power))


def netParams(model):
    '''
    Computing total network parameters
    Args:
       model: model
    return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        #print(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def main():
    
    
    print("=====> Configure dataset and pretrained model")
    configure_dataset_init_model(args)
    print(args)

    print("    current dataset:  ", args.dataset)
    print("    init model: ", args.restore_from)
    print("=====> Set GPU for training")
    if args.cuda:
        print("====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    # Select which GPU, -1 if CPU
    print("=====> Random Seed: ", args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed) 

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    print("=====> Building network")
    saved_state_dict = torch.load(args.restore_from)
    model = CoattentionNet(num_classes=args.num_classes)
    #print(model)
    new_params = model.state_dict().copy()
    calt = 0
    for i in saved_state_dict["model"]:
        #Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.') # 针对多GPU的情况
        #i_parts.pop(1)
        #print('i_parts:  ', '.'.join(i_parts[1:-1]))
        #if  not i_parts[1]=='main_classifier': #and not '.'.join(i_parts[1:-1]) == 'layer5.bottleneck' and not '.'.join(i_parts[1:-1]) == 'layer5.bn':  #init model pretrained on COCO, class name=21, layer5 is ASPP
        new_params['encoder'+'.'+'.'.join(i_parts[1:])] = saved_state_dict["model"][i]
   
    print("=====> Loading init weights,  pretrained COCO for VOC2012, and pretrained Coarse cityscapes for cityscapes")
 
            
    model.load_state_dict(new_params) #只用到resnet的第5个卷积层的参数
    #print(model.keys())
    if args.cuda:
        #model.to(device)
        if torch.cuda.device_count()>1:
            print("torch.cuda.device_count()=",torch.cuda.device_count())
            model = torch.nn.DataParallel(model).cuda()  #multi-card data parallel
        else:
            print("single GPU for training")
            model = model.cuda()  #1-card data parallel
    start_epoch=0
    
    print("=====> Whether resuming from a checkpoint, for continuing training")
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"] 
            model.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    model.train()
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    print('=====> Computing network parameters')
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    db_train = db.PairwiseImg(train=True, inputRes=input_size, db_root_dir=args.data_dir, img_root_dir=args.img_dir,  transform=None) #db_root_dir() --> '/path/to/DAVIS-2016' train path
    trainloader = data.DataLoader(db_train, batch_size= args.batch_size, shuffle=True, num_workers=0)

    optimizer = optim.SGD([{'params': get_1x_lr_params(model), 'lr': 1*args.learning_rate },  #针对特定层进行学习，有些层不学习
                {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    logFileLoc = args.snapshot_dir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t\t%s" % ('iter', 'Loss(train)\n'))
    logger.flush()

    print("=====> Begin to train")
    train_len=len(trainloader)
    print("  iteration numbers  of per epoch: ", train_len)
    print("  epoch num: ", args.maxEpoches)
    print("  max iteration: ", args.maxEpoches*train_len)
    
    for epoch in range(start_epoch, int(args.maxEpoches)):
        
        np.random.seed(args.random_seed + epoch)
        for i_iter, batch in enumerate(trainloader,0): #i_iter from 0 to len-1
            #print("i_iter=", i_iter, "epoch=", epoch)
            target, target_gt, search, search_gt, search_1, search_gt_1 = batch['target_0'], batch['target_gt_0'], batch['target_1'], batch['target_gt_1'], batch['target_2'], batch['target_gt_2']
            images, labels, images1, labels1  = batch['img'], batch['img_gt'],batch['img1'], batch['img_gt1']
            #print(labels.size())
            images.requires_grad_()
            images = Variable(images).cuda()
            labels = Variable(labels.float().unsqueeze(1)).cuda()
            images1.requires_grad_()
            images1 = Variable(images1).cuda()
            labels1 = Variable(labels1.float().unsqueeze(1)).cuda()
            
            target.requires_grad_()
            target = Variable(target).cuda()
            target_gt = Variable(target_gt.float().unsqueeze(1)).cuda()
            

            search.requires_grad_()
            search = Variable(search).cuda()
            search_gt = Variable(search_gt.float().unsqueeze(1)).cuda()

            search_1.requires_grad_()
            search_1 = Variable(search_1).cuda()
            search_gt_1 = Variable(search_gt_1.float().unsqueeze(1)).cuda()
            optimizer.zero_grad()
            
            lr = adjust_learning_rate(optimizer, i_iter+epoch*train_len, epoch,
                    max_iter = args.maxEpoches * train_len)
            #print(images.size())
            start_time = time.time()
            if i_iter%3 ==0: #对于静态图片的训练
                
                pred1, pred2, pred3, pred4, pred5 = model(images, images, images1)
                loss = 0.1*(loss_calc1(pred4, labels) + 0.8* loss_calc2(pred4, labels) + loss_calc1(pred5, labels1) + 0.8* loss_calc2(pred5, labels1) )
                loss.backward()
                
            else:

                pred1, pred2, pred3, pred4, pred5 = model(target, search, search_1)
                loss = loss_calc1(pred1, target_gt) + 0.8* loss_calc2(pred1, target_gt) + \
                       loss_calc1(pred2, search_gt) + 0.8* loss_calc2(pred2, search_gt) + \
                       loss_calc1(pred3, search_gt_1) + 0.8* loss_calc2(pred3, search_gt_1)#class_balanced_cross_entropy_loss(pred, labels, size_average=False)
                loss.backward()
            
            optimizer.step()
            end_time = time.time()
            #print('time length:', end_time-start_time)
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}  lr: {:.5f}".format(epoch, i_iter, train_len, loss, lr))
            logger.write("Epoch[{}]({}/{}):     Loss: {:.10f}      lr: {:.5f}\n".format(epoch, i_iter, train_len, loss, lr))
            logger.flush()
                
        print("=====> saving model")
        state={"epoch": epoch+1, "model": model.state_dict()}
        torch.save(state, osp.join(args.snapshot_dir, 'co_attention_'+str(args.dataset)+"_"+str(epoch)+'.pth'))


    end = timeit.default_timer()
    print( float(end-start)/3600, 'h')
    logger.write("total training time: {:.2f} h\n".format(float(end-start)/3600))
    logger.close()


if __name__ == '__main__':
    main()
