#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# print(f"GPU Number:", torch.cuda.device_count())


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms

from torch.utils.data import ConcatDataset, DataLoader, random_split
import os
import pandas as pd

from utils.sampling import iid_sample, noniid_sample
from utils.options import args_parser
from utils.Fed import *
from models.Update import LocalUpdate, LocalUpdate_pre
from models.Nets import MLP

from models.test import test_img
from models.baseline import *
from models.fedmvp import FedMVP
from models.MMFed import MMFed
# from models.transformer_networks import ViT
from sentence_transformers import SentenceTransformer, util

from utils.dataset_utils import get_cub_200_2011, get_oxford_flowers_102



if __name__ == '__main__':
    # parse args
    print("GPU Memory:", torch.cuda.memory_allocated())
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    torch.manual_seed(args.seed)
    print(f'Device:{args.device}')
    print(f'rand seed:{args.seed}')
    print(f'communication round:{args.epochs}')
    print(f'local epochs:{args.local_ep}')
    print(f'dataset:{args.dataset}')
    print(f'model:{args.model}')
    print(f'pretrained:{args.pretrain}')
    print(f'training batch size:{args.local_bs}')
    print(f'comment:{args.comment}')
    print(f'learning rate:{args.lr}')
    print(f'missing ratio:{args.missing_ratio}')
    print(f'image to text missing ratio:{args.img_missing_ratio}:{1.0-args.img_missing_ratio}')


    # load dataset and split users
    if args.dataset == 'cub': # total 11788 images
        
        # # 8855 train images
        # train_set, train_loader = get_cub_200_2011(split='train_val', d_batch=args.local_bs)
        # # 2933 test images
        # test_set, test_loader = get_cub_200_2011(split='test', d_batch=args.bs)
        
        # if args.missing_ratio != 1.0:

        
        # 10610 train images
        train_set, train_loader = get_cub_200_2011(split='train_val', d_batch=args.local_bs)
        # 1178 test images
        test_set, test_loader = get_cub_200_2011(split='test', d_batch=args.bs)
        
        # total_set = ConcatDataset([train_set_, test_set_])

        # total_loader = DataLoader(total_set, batch_size=args.local_bs, shuffle=True, drop_last=True, num_workers=4,
                                #   pin_memory=True)
        if args.missing_ratio < 1.0 and args.model != 'fedmvp':
            train_set, _set = random_split(train_set, [1.0-args.missing_ratio, args.missing_ratio])
            
        # train_set, test_set = random_split(total_set, [0.9*(1-args.missing_ratio), 1-0.9*(1-args.missing_ratio)])
        
        # train_loader = DataLoader(train_set, batch_size=args.local_bs, shuffle=True, drop_last=True, num_workers=4,
                                #   pin_memory=True)

        # test_loader = DataLoader(test_set, batch_size=args.local_bs, shuffle=True, drop_last=True, num_workers=4,
                                #  pin_memory=True)

        # sample users
        if args.iid:
            dict_users = iid_sample(train_set, args.num_users)
        else:
            dict_users = torch.load('/home/lfc5481/project/MCFL/data/cub/noniid.pth')

    elif args.dataset == 'flower': # 8189


        # 7370 train images
        train_set, train_loader = get_oxford_flowers_102(split='train_val', d_batch=args.local_bs)
        # 819 test images
        test_set, test_loader = get_oxford_flowers_102(split='test', d_batch=args.bs)
        
        if args.missing_ratio < 1.0 and args.model != 'fedmvp':
            train_set, _set = random_split(train_set, [1.0-args.missing_ratio, args.missing_ratio])
        
        # print("GPU Memory:", torch.cuda.memory_allocated())

        if args.iid:
            dict_users = iid_sample(train_set, args.num_users)
        else:
            dict_users = noniid_sample(train_set, args.num_users, args)

    else:
        exit('Error: unrecognized dataset')
   

    img_size = 256
    print(f'image size:{img_size}')

    print('dataset is loaded')
    print(f'train set size:{len(train_set)}')
    print(f'test set size:{len(test_set)}')

    
    MODEL_SAVE_PATH = './save/trained_model/'
    BERT_CUB_PATH = os.path.join(MODEL_SAVE_PATH, 'fedbert_cub_iidTrue.pth')
    BERT_FLOWER_PATH = os.path.join(MODEL_SAVE_PATH, 'fedbert_flower_iidTrue.pth')
    
    
    # build model
    if args.model == 'fedclip':

        encoder = SentenceTransformer('clip-ViT-B-32')
        net_glob = MLP(dim_in=512*2, dim_hidden1=1024, dim_hidden2=512, dim_out=args.num_classes)
    elif args.model == 'fedrn50':
        net_glob = FedRn50(args)

    elif args.model == 'fedvit':
        net_glob = FedViT(args=args, image_size=int(img_size))
        
    elif args.model == 'fedbert':
        net_glob = FedBERT(args=args)
        
        if args.dataset == 'cub':
            bert_para = torch.load(BERT_CUB_PATH)
        else:
            bert_para = torch.load(BERT_FLOWER_PATH)
                
        net_glob.load_state_dict(bert_para)
        

    elif args.model == 'mmfed':
        net_glob = MMFed(embed_dim=256, num_heads=4, num_classes=args.num_classes, num_layers=2, text_dim=256,
                        image_dim=img_size)


    elif args.model == 'fedmvp':
        
        net_glob = FedMVP(args=args, loss_type='all').to(args.device)
        
    else:
        exit('Error: unrecognized model')

    print('Global model is loaded')

    print(net_glob)
    net_glob.train()
    


    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    log_loss = []

    print('=======Start Training!=======')
    # if args.all_clients: 
    #     print("Aggregation over all clients")
    #     w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        # if not args.all_clients:
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print('Client list', idxs_users)
        for idx in idxs_users:

            local = LocalUpdate(args=args, dataset=train_set, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            
            # if args.all_clients:
            #     w_locals[idx] = copy.deepcopy(w)
            # else: 
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        if args.model == 'fedmvp':
            cka_graph = cka_graph(net_g=net_glob, client_paras=w_locals)
            w_glob = cka_aggregation(cclient_params_list=w_locals,cka_matrix=cka_graph)
        else:
            w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)

        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # save loss log
        if iter % 30 == 0:
            net_glob.eval()

            acc_train, loss_ = test_img(net_glob, train_set, args, dataset_type='Train')
            acc_test, loss_t = test_img(net_glob, test_set, args)
            print(f'Round{iter}, Global Train Acc:{acc_train:.3f}')
            print(f'Round{iter}, Global Test Acc:{acc_test:.3f}')

            if args.verbose:
                for client_idx, local_w in enumerate(w_locals):
                    net_glob.load_state_dict(local_w)
                    acc_test, loss_t = test_img(net_glob, test_set, args)
                    print(f'Round{iter}, Client-{client_idx} Test Acc:{acc_test:.3f}')


            net_glob.load_state_dict(w_glob)
            net_glob.train()

        else:
            acc_train = -1
            acc_test = -1
        log_loss.append([iter, loss_avg, acc_train, acc_test])
        df = pd.DataFrame(log_loss, columns=['round', 'loss', 'acc_train', 'acc_test'])
        df.to_excel(
            './save/fed_{}_{}_{}_C{}_pretrain{}_{}.xlsx'.format(args.dataset, args.model, args.epochs, args.frac,
                                                                args.pretrain, args.comment), index=False)

    print('======Training Finished!======')

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_frac{}_{}_Comm{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.missing_ratio, args.comment))

    # testing
    net_glob.eval()
    
    
    torch.save(net_glob.state_dict(), './save/trained_model/{}_{}_iid{}.pth'.format(args.model, args.dataset, args.iid))



    print('======evaluation begin======')

    acc_train, loss_train = test_img(net_glob, train_set, args, dataset_type='Train')
    acc_test, loss_test = test_img(net_glob, test_set, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    print('=======END========')
    log_loss.append([args.epochs, loss_test, acc_train, acc_test])
    df = pd.DataFrame(log_loss, columns=['round', 'loss', 'acc_train', 'acc_test'])
    df.to_excel('./save/fed_{}_{}_{}_C{}_pretrain{}_{}.xlsx'.format(args.dataset, args.model, args.epochs, args.frac,
                                                                    args.pretrain, args.comment), index=False)