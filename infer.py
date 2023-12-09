# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 00:49:18 2023

@author: Arino Jenynof
"""
from pathlib import Path
import numpy as np
import torch
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from dataset import Sorghum100DataSet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = Path("result-fullset-best.pth").resolve()
    state_dict = torch.load(f)

    # model = timm.create_model("resnetrs152", pretrained=True, num_classes=100, global_pool="catavgmax", drop_rate=0.25)
    model = timm.create_model("tf_efficientnet_b5_ns", pretrained=True, num_classes=100, global_pool="catavgmax", drop_rate=0.25)
    img_h, img_w = model.pretrained_cfg["input_size"][1:]
    model.load_state_dict(state_dict)
    model.to(device)

    test_transform = A.Compose([
        A.CLAHE(40.0, (16, 16), p=1.0),
        A.Resize(img_h, img_w, cv2.INTER_CUBIC),
        A.Normalize(),
        ToTensorV2()
    ])

    root = Path("../small-jpegs-fgvc").resolve()
    test_ds = Sorghum100DataSet(root, test_transform, is_train=False)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=16, num_workers=12, pin_memory=True)

    model.eval()
    res = []
    with torch.no_grad():
        for images in tqdm(test_dl):
            images = images.to(device)
            with torch.autocast("cuda"):
                preds = model(images)

            res += preds.argmax(1).tolist()

    df = test_ds.df_test.copy()
    df.loc[:, "cultivar"] = res
    for key, val in test_ds.idx_to_class.items():
        df.loc[df["cultivar"] == key, "cultivar"] = val
    df.to_csv(f"{f.stem}.csv", index=False)