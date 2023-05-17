#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import argparse
from utils.options import args_parser

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        for i in range(1, len(w)):
            # print(f'client:{k}, key:{i}')
            # print(w_avg[k].device)
            # print(w[i][k].device)
            w[i][k].to(w_avg[k].device)
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


