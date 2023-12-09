# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 00:01:41 2023

@author: Arino Jenynof
"""
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import PIL

class Sorghum100DataSet(Dataset):
    def __init__(self, root: Path, transform, is_train=True):
        if is_train:
            self.image_path = root / "train"
        else:
            self.image_path = root / "test"
        self.is_train = is_train

        # Create idx_to_class and class_to_idx like in pytorch
        self.df = pd.read_csv(root / "train_cultivar_mapping.csv")
        self.df = self.df.drop(["file_path", "is_exist"], axis=1)
        self.df["class_to_idx"] = pd.Series(data=[0 for _ in range(self.df.shape[0])], dtype="Int64")
        self.idx_to_class = dict()
        for i, label in enumerate(self.df["cultivar"].unique()):
            self.idx_to_class[i] = label
            self.df.loc[self.df["cultivar"] == label, "class_to_idx"] = i

        # Albumentations.Compose, not pytorch.Compose
        self.transform = transform
        
        # Test image
        self.df_test = pd.read_csv(root / "sample_submission.csv")

    def __len__(self):
        if self.is_train:
            return self.df.shape[0]
        else:
            return self.df_test.shape[0]

    def __getitem__(self, index):
        if self.is_train:
            image = PIL.Image.open(self.image_path / self.df.iloc[index, 0])
            target = self.df.iloc[index, 2]
            image = self.transform(image=np.array(image))["image"]
            return image, target
        else:
            png = self.image_path / self.df_test.iloc[index, 0]
            image = PIL.Image.open(png.parent / (png.stem + ".jpeg"))
            image = self.transform(image=np.array(image))["image"]
            return image