#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def iid_sample(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users) # number of data samples per client
    # all_idxs: 0 to len(dataset) indexs of the data samples not their ids in the dataset
    dict_users, all_idxs = {}, [i for i in range(len(dataset))] 
    for i in range(num_users):
        # random sampling by num_items without replacement
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid_sample(dataset, args):
    """
    cub 200: train set:10610, class:200
    flower 102: train set: 7370 class: 102
    Fact1: 
    
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:cub or flower
    :param num_users: fixed 10
    :return:
    """
    if args.dataset == 'cub':
        total_num = 10610
        num_shards = 200
    else:
        total_num = 7370
        num_shards = 102
    
    num_users = 10
    num_imgs = int(len(dataset)/num_shards)
    
    
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


