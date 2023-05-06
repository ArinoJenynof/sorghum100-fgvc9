# Sorghum-100 Cultivar Identification - FGVC 9 [Kaggle Link](https://www.kaggle.com/competitions/sorghum-id-fgvc-9/)
Identify sorghum varietals. Done for my master's study's homework, failed to meet demanded accuracy. Uploaded here for archival purpose, and maybe for somebody else to use.

# Components
Splitted for exactly this archival purpose. Originally written in one file of a very, very hot mess.
1. Sorghum-100 dataset wrapped in `VisionDataset`. Trying to mimic other TorchVision data set.
2. Model for classification. Mix EfficientNet with ArcFace. Leverage `timm` heavily.
3. Training script. A bad imitation of ImageNet/timm training script.
4. Inference script. Outputs a csv just like the competition wanted.

# Step-by-step
Or how to run this s***
1. Make sure to install [PyTorch](https://pytorch.org/get-started/locally/) and [timm](https://github.com/huggingface/pytorch-image-models) first
2. Download Sorghum-100 [data set](https://www.kaggle.com/competitions/sorghum-id-fgvc-9/data) and put it into `data` folder
3. `python train.py & python infer.py`
4. ...
5. ~~PROFIT!~~ Nah this model only got `0.733` in private and `0.748` public score