#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataset_utils import to_one_hot
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

def test_img(net_g, datatest, args, dataset_type='Test'):
    net_g.to(args.device)
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (images, labels, texts) in enumerate(data_loader):

        if args.model == 'fedbert':

            labels = labels.to(args.device)

            log_probs = net_g(texts[0].to(args.device), texts[1].to(args.device), texts[2].to(args.device))
            log_probs = log_probs.to(args.device)
            test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()

        elif args.model == 'fedvit' or args.model == 'fedrn50':
            images, labels= images.to(args.device), labels.to(args.device)
            log_probs = net_g(images)
            log_probs = log_probs.to(args.device)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()

        elif args.model == 'fedmvp':

            labels_oh = to_one_hot(labels, args.num_classes)
            labels = labels.to(args.device)
            log_probs, batch_representation = net_g(
                images.to(args.device), 
                texts[0].to(args.device), 
                texts[1].to(args.device), 
                texts[2].to(args.device)
                )
            loss = net_g.training_loss(log_probs, batch_representation, labels)
            test_loss += loss.item()
            log_probs = log_probs.to(args.device)

        else:

            images, labels, texts = images.to(args.device), labels.to(args.device), texts.to(args.device)
            # images, labels = images.to(args.device), labels.to(args.device)
            net_g.zero_grad()
            log_probs = net_g(images, texts)

            # log_probs = torch.softmax(net_g(images),dim=0)
            log_probs = log_probs.to(args.device)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
            # get the index of the max log-probability


        y_pred = log_probs.data.max(1, keepdim=True)[1]

        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\n{} set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(dataset_type,
            test_loss, correct, len(data_loader.dataset), accuracy))
    net_g.train()
    return accuracy, test_loss


