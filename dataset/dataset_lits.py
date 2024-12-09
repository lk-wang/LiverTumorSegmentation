
import os
import numpy as np
import cv2  #https://www.jianshu.com/p/f2e88197e81d
import random
from skimage.io import imread
from skimage import color
import torch
import torch.utils.data
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from scipy import ndimage
from skimage import color




def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


# 转换成tensor，Normalize[0，1]之间的数据投射到[-1,1]
# .ToTensor(),将(H x W x C)形状转换为 (C x H x W) 的tensor。还会将数值从 [0, 255] 归一化到[0,1]还会将数值从 [0, 255] 归一化到[0,1]
transform = transforms.Compose([

    transforms.ToTensor()     #  将(H x W x C)形状转换为 (C x H x W) 的tensor。还会将数值从 [0, 255] 归一化到[0,1]还会将数值从 [0, 255] 归一化到[0,1]
    
])


def data_in_one(inputdata):
    min = np.nanmin(inputdata)
    max = np.nanmax(inputdata)
    outputdata = (inputdata-min)/(max-min)
    return outputdata



class MyDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,list_name,transform=transform,aug=False):  # data_path,txt文件地址

        self.filename_list = self.load_file_name_list(os.path.join(data_path, list_name))
        self.aug=aug
        self.transform = transform
 
        
    def augment(self, image, mask):
        if random.random() > 0.5:
            image, mask = random_rot_flip(image, mask)
        elif random.random() > 0.5:
            image, mask = random_rotate(image, mask)
        return image,mask

    def __getitem__(self, index):

        img_path  = self.filename_list[index][0]
        mask_path = self.filename_list[index][1]
        
        img = cv2.imread(img_path)   # 读取ct的路径
        mask = cv2.imread(mask_path,cv2.COLOR_BGR2GRAY)   # 3通道转换成1通道
          
        if self.aug is True:
            img,mask=self.augment(img,mask) 
            
        #img = cv2.resize(img, (256, 256))  # (256,256,3)
        #mask = cv2.resize(mask, (256, 256))  # (256,256)
        #mask[mask>0]=255
       
        img = self.transform(img)
        mask = self.transform(mask)
        
        return img, mask,img_path,mask_path

    def __len__(self):
        return len(self.filename_list)


    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
    # 定义数据加载
    path='/home/data/WB_/LiverSegProject/data/LITS'
   
    ds=MyDataset(data_path=path,list_name='train_path_list.txt')
    
    train_size = int(0.8 * len(ds))  # 整个训练集中，百分之90为训练集
    val_size = len(ds) - train_size
    train_dataset, val_dataset = random_split(ds, [train_size, val_size])  # 划分训练集和验证集

    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)  #
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)  #

    #for i, (ct, seg,ct_path,seg_path) in enumerate(train_dl):
        #print(i, img_path, mask_path)
        #print(i,ct.size(),seg.size())   # 4600 torch.Size([1, 3, 448, 448]) torch.Size([1, 1, 448, 448])
        #print(i,ct_path,seg_path)
    for i, (ct, seg,ct_path,seg_path) in enumerate(val_dl):
        #print(i, img_path, mask_path)
        print(i,ct.size(),seg.size())   # 1150 torch.Size([1, 3, 448, 448]) torch.Size([1, 1, 448, 448])
        #print(i,ct_path,seg_path)


        
    