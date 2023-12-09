# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 00:30:07 2023

@author: Arino Jenynof
"""
import random
from pathlib import Path
import numpy
from sklearn.model_selection import train_test_split
import torch
import timm
import timm.optim
import timm.scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from dataset import Sorghum100DataSet
from model import Sorghum100Model

if __name__ == "__main__":
    random.seed(42069)
    numpy.random.seed(42069)
    torch.cuda.manual_seed(42069)
    torch.cuda.manual_seed_all(42069)
    torch.manual_seed(42069)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 45
    
    # model = timm.create_model("resnetrs152", pretrained=True)
    model = timm.create_model("tf_efficientnet_b6_ns", pretrained=True)
    img_h, img_w = model.pretrained_cfg["input_size"][1:]
    
    train_transform = A.Compose([
        A.CLAHE(40.0, (16, 16), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(0.2, 0.2, 20, cv2.INTER_CUBIC, p=0.5),
        A.OneOf([A.RandomBrightnessContrast(0.1, 0, p=1.0),
                 A.RandomBrightnessContrast(0, 0.1, p=1.0)]),
        A.RandomResizedCrop(img_h, img_w, interpolation=cv2.INTER_CUBIC),
        A.Normalize(),
        ToTensorV2(),
    ])
    
    valid_transform = A.Compose([
        A.CLAHE(40.0, (16, 16), p=1.0),
        A.Resize(img_h, img_w, cv2.INTER_CUBIC),
        A.Normalize(),
        ToTensorV2()
    ])
    
    root = Path("../small-jpegs-fgvc").resolve()
    train_ds = Sorghum100DataSet(root, train_transform)
    valid_ds = Sorghum100DataSet(root, valid_transform)
    
    train_idx, valid_idx = train_test_split(
        [i for i in range(len(train_ds))],
        test_size=0.05,
        shuffle=True,
        stratify=train_ds.df["class_to_idx"],
        random_state=42069
    )
    
    # model = timm.create_model("resnetrs152", pretrained=True, num_classes=100, global_pool="catavgmax", drop_rate=0.25)
    # model = timm.create_model("tf_efficientnet_b5_ns", pretrained=True, num_classes=100, global_pool="catavgmax", drop_rate=0.25)
    model = Sorghum100Model()
    img_h, img_w = model.backbone.pretrained_cfg["input_size"][1:]
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimiser = timm.optim.create_optimizer_v2(model, opt="AdamW", lr=0.001, weight_decay=0.001)
    scheduler = timm.scheduler.CosineLRScheduler(optimiser, t_initial=45, warmup_t=10, lr_min=0.00001)

    train_sub = torch.utils.data.Subset(train_ds, train_idx)
    valid_sub = torch.utils.data.Subset(valid_ds, valid_idx)

    train_dl = torch.utils.data.DataLoader(train_sub, batch_size=16, num_workers=12, pin_memory=True)
    valid_dl = torch.utils.data.DataLoader(valid_sub, batch_size=16, num_workers=12, pin_memory=True)

    res_path = Path("./result-fullset-arcface.tar").resolve()
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        scaler = torch.cuda.amp.GradScaler()
        num_batches = len(train_dl)
        num_updates = epoch * num_batches
        running_loss, running_corrects = 0.0, 0.0

        for images, labels in tqdm(train_dl):
            images = images.to(device)
            labels = labels.to(device)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                preds = model(images, labels)
                loss = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            num_updates += 1
            scheduler.step_update(num_updates)

            running_loss += loss.item() * images.size(0)
            running_corrects += (preds.argmax(1) == labels).double().sum().item()
        scheduler.step(epoch + 1)

        train_l = running_loss / (len(train_dl.dataset))
        train_a = running_corrects / (len(train_dl.dataset))

        print(f"train_loss {train_l}, train_acc {train_a}")
        train_loss.append(train_l)
        train_acc.append(train_a)

        model.eval()
        running_loss, running_corrects = 0.0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(valid_dl):
                images = images.to(device)
                labels = labels.to(device)

                with torch.autocast("cuda"):
                    preds = model(images, labels)
                    loss = criterion(preds, labels)

                running_loss += loss.item() * images.size(0)
                running_corrects += (preds.argmax(1) == labels).double().sum().item()
        valid_l = running_loss / (len(valid_dl.dataset))
        valid_a = running_corrects / (len(valid_dl.dataset))

        print(f"valid_loss {valid_l}, valid_acc {valid_a}")
        valid_loss.append(valid_l)
        valid_acc.append(valid_a)

        if valid_a > best_acc:
            best_acc = valid_a
            torch.save(model.state_dict(), res_path.parent / (res_path.stem + "-best.pth"))
    torch.save({
        "state_dict": model.state_dict(),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc
    }, res_path)