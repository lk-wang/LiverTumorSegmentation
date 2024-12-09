import numpy as np
import cv2  # https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import os

transform = transforms.Compose([
    transforms.ToTensor()  # 将(H x W x C)形状转换为 (C x H x W) 的tensor。还会将数值从 [0, 255] 归一化到[0,1]还会将数值从 [0, 255] 归一化到[0,1]
])


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, img_path, mask_path, transform=transform, aug=False):
        self.transform = transform
        self.aug = aug
        self.imgs = []
        self.masks = []
        for name in os.listdir(img_path):
            img = os.path.join(img_path, name)  # 99-566_ct.png
            mask = os.path.join(mask_path, name.replace("ct", "tumor"))  # 99-566_tumor.png
            self.imgs.append(img)
            self.masks.append(mask)

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，flip Code为 1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        # 读numpy数据(npy)的代码
        img = cv2.imread(img_path)  # 读取ct的路径
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)  # 3通道转换成1通道

        # 随机进行数据增强，为2时不做处理 ,1 水平翻转，0垂直翻转，-1 水平+垂直翻转
        if self.aug == True:
            flipCode = random.choice([-1, 0, 1, 2])
            if flipCode != 2:
                img = self.augment(img, flipCode)
                mask = self.augment(mask, flipCode)

        img = self.transform(img)
        mask = self.transform(mask)

        return img, mask, img_path, mask_path


# 生成的样本和标签为[batch_size,3,448,448]和[batch_size,448,448]
img_path = 'E:/datasets/3DIRCADB_PNG/3DIRCADB_PNG_without_no-null/Tumor/image'
mask_path = 'E:/datasets/3DIRCADB_PNG/3DIRCADB_PNG_without_no-null/Tumor/mask'
ds = MyDataset(img_path, mask_path, aug=True)
torch.manual_seed(0)
train_size = int(0.6 * len(ds))  # 整个训练集中，百分之80为训练集
val_size = int(0.2 * len(ds))
test_size = len(ds) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(ds, [train_size, val_size, test_size])  # 划分训练集和验证集

if __name__ == '__main__':
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)  #
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)  #
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)  #

    print('训练集总数：', len(train_dataset))
    print('验证集总数：', len(val_dataset))
    print('测试集总数：', len(test_dataset))

    print('数据集总数：', len(train_dataset)+len(val_dataset)+len(test_dataset))

    for i, (image, label, image_path, label_path) in enumerate(test_dataloader):
        # print(i, image.size(), label.size())
        print(image_path)
