#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from torch.nn import functional as F

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



def cka_aggregation(client_params_list, cka_matrix):
    
    # Compute the graph importance for each client
    graph_importance = cka_matrix.sum(axis=1)
    # Normalize the graph importance
    normalized_graph_importance = graph_importance / graph_importance.sum()

    # Initialize the global model parameters with zeros
    global_params = [torch.zeros_like(param) for param in client_params_list[0]]

    # Perform the weighted sum of client parameters
    for i, client_params in enumerate(client_params_list):
        weight = normalized_graph_importance[i]
        for j, param in enumerate(client_params):
            global_params[j] += weight * param

    return global_params

def cka_similarity(x1, x2):
    
    # Reshape the input tensors to have shape (batch_size, -1)
    x1 = x1.view(x1.size(0), -1)
    x2 = x2.view(x2.size(0), -1)
    
    # normalize 
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)

    # Compute the centered feature maps
    mean1 = torch.mean(x1, dim=0, keepdim=True)
    mean2 = torch.mean(x2, dim=0, keepdim=True)
    centered1 = x1 - mean1
    centered2 = x2 - mean2

    # Compute the CKA similarity
    gram1 = torch.matmul(centered1, centered1.T)
    gram2 = torch.matmul(centered2, centered2.T)
    cka = torch.trace(torch.matmul(gram1, gram2)) / \
          torch.sqrt(torch.trace(torch.matmul(gram1, gram1))) / \
          torch.sqrt(torch.trace(torch.matmul(gram2, gram2)))

    return cka.item()


CKA_CUB = '/home/lfc5481/project/MCFL/data/cub/cka_cub.pth'
CKA_FLOWER = '/home/lfc5481/project/MCFL/data/flower_data/cka_flower.pth'

CUB_META = '/home/lfc5481/project/MCFL/data/cub/Metadata.pth'
FLOWER_META = '/home/lfc5481/project/MCFL/data/flower_data/metadata.pth'

def cka_graph(net_g, client_paras, image, text):
    
    similarity_matrix = torch.zeros((len(client_paras), len(client_paras)))
    # Compute the CKA similarity between the learned representations of the models
    
    for i, model1_para in enumerate(client_paras):
        model1 = net_g.load_state_dict(model1_para)
        for j, model2_para in enumerate(client_paras):
            if i == j:
                similarity_matrix[i][j] = 1.0
                continue
            
            model2 = net_g.load_state_dict(model2_para)
            with torch.no_grad():
                representations1 = model1(image, text).last_hidden_state
                representations2 = model2(image, text).last_hidden_state
                representations1 = torch.cat(representations1, dim=0)
                representations2 = torch.cat(representations2, dim=0)
            similarity = cka_similarity(representations1, representations2)
            similarity_matrix[i][j] = similarity
    
    return similarity_matrix



if __name__ == '__main__':
    
    pseudo_img = torch.rand((64,256))
    pseudo_text = torch.rand((64,256))
    
    cka_sim = cka_similarity(pseudo_img, pseudo_text)
    print(f'cka_sim:{cka_sim}')