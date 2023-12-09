# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 23:58:13 2023

@author: Arino Jenynof
"""
import math
import torch
from torch.nn import functional as F

class ArcMarginProduct(torch.nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        
        self.weight = torch.nn.Parameter(torch.zeros([out_features, in_features]))
        torch.nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        
    def forward(self, inputs, labels):
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = (cosine * self.cos_m) - (sine * self.sin_m)
        
        phi = torch.where(cosine > self.theta, phi, cosine - self.sinmm)
        
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        out = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        out *= self.scale
        return out