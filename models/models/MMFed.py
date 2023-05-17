#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8


import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from utils.options import args_parser
from transformers import BertModel

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')



class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)




class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_heads = num_heads
        # input_dim must be divisible by num_heads
        self.head_dim = input_dim // num_heads

        self.scaling = math.sqrt(self.head_dim)

        # input linear layer
        self.input_projector_weight = Parameter(torch.Tensor(3 * input_dim, input_dim))

        self.register_parameter('input_projector_weight', self.input_projector_weight)
        self.input_projector_bias = Parameter(torch.Tensor(3 * input_dim))
        # output linear layer
        self.output_projector = nn.Linear(input_dim, output_dim, bias=True)

        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.input_projector_weight)
        nn.init.xavier_uniform_(self.output_projector.weight)
        nn.init.constant_(self.input_projector_bias, 0.)
        nn.init.constant_(self.output_projector.bias, 0.)


    def forward(self, input, guide_input = None):
        # Input shape: Sequence length x Batch x Channel

        seq_len, bs, embed_dim = input.size()

        if guide_input == None:
            # self-attention
            q, k, v = self.qkv_projector(input)
        else:
            # guided attention
            q = self.query_projector(guide_input)
            k, v = self.kv_projector(input)

        # reshape
        q = q.contiguous().view(seq_len, bs * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        seq_len_v = v.size(1)

        # compute attention weights: softmax((q * k.T)/ sqrt(embed_dim))
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = attn_weights / self.scaling


        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)


        # compute the attention
        attn = torch.matmul(attn_weights, v)

        # reshape the weighted attention result
        attn = attn.transpose(0, 1).contiguous().view(seq_len, bs, embed_dim)
        attn = self.output_projector(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bs, self.num_heads, seq_len, seq_len_v)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def qkv_projector(self, query):
        return self.projector(query).chunk(3, dim=-1)

    def kv_projector(self, key):
        return self.projector(key, start=self.input_dim).chunk(2, dim=-1)

    def query_projector(self, query, **kwargs):
        return self.projector(query, end=self.input_dim, **kwargs)


    def projector(self, input, start=0, end=None):
        weight = self.input_projector_weight
        bias = self.input_projector_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)




class MMFed(nn.Module):

    def __init__(self, embed_dim, num_heads, num_classes, num_layers, image_dim, text_dim):
        super().__init__()

        self.embed_dim = embed_dim
        
        self.text_embedding = BertModel.from_pretrained('bert-base-uncased')
        for p in self.parameters():
            p.requires_grad = False
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.num_classes = num_classes
        self.patch_embedding = SPT(dim = image_dim, patch_size = 16, channels = 3)

        # self.text_embedding = nn.Embedding(num_embeddings=8418, embedding_dim=256)
        
        self.text_mapping = nn.Linear(768,self.embed_dim)
        # self.text_input_mapping = nn.Linear(768,common_dim)


        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            new_layer = CoAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
            self.layers.append(new_layer)

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(4 * self.embed_dim, self.num_classes),
            nn.Dropout(p=0.1),
        )



    def forward(self, image, text, attention_mask, token_id):
        """
        input: (seq_len, batch size, embed_dim)
        """

        mod1 = self.patch_embedding(image) # (64,196,256)
        mod2_ = self.text_embedding(text, 
                                    attention_mask=attention_mask, 
                                    token_type_ids=token_id).last_hidden_state # (64,196,768)
        # pooler_output (64,768)
        mod2 = self.text_mapping(mod2_)
        # mod2 = mod2[0]

        # Add positional embedding
        mod1 += self.embed_positions(mod1.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        mod2 += self.embed_positions(mod2.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        # mod1 += self.embed_positions(mod1.transpose(0, 1)).transpose(0, 1) # (64,196,256)
        # mod2 += self.embed_positions(mod2.transpose(0, 1)).transpose(0, 1)


        # encoder layers
        intermediates = [mod1, mod2]
        for layer in self.layers:

            mod1, mod2 = layer(mod1, mod2)
            mod1 = self.layer_norm(mod1)
            mo2 = self.layer_norm(mod2)
            intermediates.append(mod1)
            intermediates.append(mod2)

        output = torch.cat((mod1, mod2),dim=-1)
        output = output.mean(dim=1)

        return self.classifier(output)



class CoAttentionBlock(nn.Module):

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attention1 = MultiheadAttention(
            input_dim=self.embed_dim,
            output_dim=self.embed_dim,
            num_heads=self.num_heads,
        )

        self.self_attention2 = MultiheadAttention(
            input_dim=self.embed_dim,
            output_dim=self.embed_dim,
            num_heads=self.num_heads,
        )

        self.guide_attention = MultiheadAttention(
            input_dim=self.embed_dim,
            output_dim=self.embed_dim,
            num_heads=self.num_heads,
        )

        self.Linear = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
            nn.Dropout(p=0.1),
        )

        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])


    def forward(self, mod1, mod2):
        residual1 = mod1
        residual2 = mod2

        x1 = self.layer_norms[0](mod1)
        x2 = self.layer_norms[0](mod2)

        x1, _ = self.self_attention1(input=x1)
        x2, _ = self.self_attention2(input=x2)

        x1 = F.dropout(x1, p=0.1, training=self.training)
        x2 = F.dropout(x2, p=0.1, training=self.training)
        x1 = residual1 + x1
        x2 = residual2 + x2

        x1,_ = self.guide_attention(input=x1, guide_input=x2)

        residual1 = x1
        residual2 = x2

        x1 = self.layer_norms[1](x1)
        x2 = self.layer_norms[1](x2)

        x1 = self.Linear(x1)
        x2 = self.Linear(x2)
        x1 = residual1 + x1
        x2 = residual2 + x2

        return x1, x2








# Code adapted from the fairseq repo. https://github.com/facebookresearch/fairseq

def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    if getattr(make_positions, buf_name).numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()   # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).reshape(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.reshape(-1)).reshape(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number