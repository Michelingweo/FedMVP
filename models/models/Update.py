#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from utils.dataset_utils import to_one_hot
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from sklearn import metrics
import transformers
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from utils.options import args_parser


args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label, text = self.dataset[self.idxs[item]]
        return image, label, text


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.to(args.device)
        # train and update
        optimizer = optim.AdamW(net.parameters(), lr=args.lr)

        # scheduler1 = transformers.get_linear_schedule_with_warmup(optimizer, int(epochs * 0.1), epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=int(args.local_ep))
        scaler = GradScaler()

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            net.train()
            for batch_idx, (images, labels, texts) in enumerate(self.ldr_train):
                labels_oh = to_one_hot(labels, args.num_classes)

                if args.model == 'fedmvp':
                    optimizer.zero_grad()
                    with autocast():
                        output, batch_representation = net.forward(
                            images.to(args.device), 
                            texts[0].to(args.device), 
                            texts[1].to(args.device), 
                            texts[2].to(args.device)
                            )
                        loss = net.training_loss(output, batch_representation, labels.to(args.device))
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    
                elif args.model == 'fedvit' or args.model == 'fedrn50':
                    images, labels= images.to(args.device), labels.to(args.device)

                    net.zero_grad()
                    with autocast():
                        log_probs = net(images)
                        # log_probs = torch.softmax(net(images), dim=0)
                        loss = self.loss_func(log_probs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # loss.backward()
                    # optimizer.step()

                elif args.model == 'fedbert':

                    optimizer.zero_grad()

                    labels = labels.to(args.device)
                    net.zero_grad()
                    with autocast():
                        log_probs = net(texts[0].to(args.device), texts[1].to(args.device), texts[2].to(args.device))

                        # loss = self.loss_func(log_probs, labels)
                        loss = F.cross_entropy(log_probs, labels_oh.to(args.device))
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # loss.backward()
                    # optimizer.step()
                else:

                    images, labels, texts = images.to(args.device), labels.to(args.device), texts.to(args.device)

                    net.zero_grad()
                    with autocast():

                        log_probs = net(images, texts)
                    # log_probs = torch.softmax(net(images), dim=0)
                        loss = self.loss_func(log_probs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # loss.backward()
                    # optimizer.step()


                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

                scheduler.step()

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)



class LocalUpdate_pre(object):
    def __init__(self, args, dataset=None, idxs=None, encoder=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.encoder = encoder
    def train(self, net):
        net.train()
        encoder = self.encoder.to(args.device)
        encoder.eval()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer = Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, texts) in enumerate(self.ldr_train):
                # labels = to_one_hot(labels, args.num_classes)
                if args.model == 'fedViT':
                    images, labels = images.to(args.device), labels.to(args.device)
                    embedding = encoder(images)
                    net.zero_grad()
                    log_probs = net(embedding)
                    # log_probs = torch.softmax(net(images), dim=0)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                elif args.model == 'fedBERT':
                    texts = [list(text) for text in texts]
                    texts = torch.tensor(texts)
                    texts = texts.transpose(1, 0)
                    texts, labels = texts.to(args.device), labels.to(args.device)
                    embedding = self.encoder(texts)
                    net.zero_grad()
                    log_probs = net(embedding)
                    # log_probs = torch.softmax(net(images), dim=0)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                elif args.model == 'fedClip':
                    labels = labels.to(args.device)
                    embedding = encoder(images, texts)
                    embedding = torch.tensor(embedding).to(args.device)
                    net.zero_grad()
                    log_probs = net(embedding)
                    # log_probs = torch.softmax(net(images), dim=0)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                elif args.model == 'mcfed':
                    texts = [list(text) for text in texts]
                    texts = torch.tensor(texts)
                    texts = texts.transpose(1, 0)
                    images, labels, texts = images.to(args.device), labels.to(args.device), texts.to(args.device)

                    net.zero_grad()
                    loss, _ = net.training_step(images, texts, labels)
                    loss.backward()
                    optimizer.step()

                else:
                    texts = [list(text) for text in texts]
                    texts = torch.tensor(texts)
                    texts = texts.transpose(1, 0)
                    images, labels, texts = images.to(args.device), labels.to(args.device), texts.to(args.device)
                    embedding = encoder(images, texts)

                    net.zero_grad()
                    log_probs = net(embedding)
                    # log_probs = torch.softmax(net(images), dim=0)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)



