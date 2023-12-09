# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:26:55 2023

@author: Arino Jenynof
"""
import torch
import timm
from arcface import ArcMarginProduct

class Sorghum100Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b5_ns", pretrained=True, num_classes=100, global_pool="catavgmax", drop_rate=0.25)
        in_features = self.backbone.get_classifier().in_features
        out_features = self.backbone.get_classifier().out_features
        self.backbone.reset_classifier(0, "catavgmax")
        
        self.last_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.PReLU()
        )
        
        self.classifier = ArcMarginProduct(1024, out_features)
        
    def forward(self, images, labels):
        ft = self.backbone(images)
        ft = self.last_layer(ft)
        preds = self.classifier(ft, labels)
        return preds
    
    def forward_features(self, images):
        ft = self.backbone(images)
        ft = self.last_layer(ft)
        return ft