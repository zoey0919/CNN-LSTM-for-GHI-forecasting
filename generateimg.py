# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:57:54 2022

@author: 123
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
# 将数据变化为张量
import pandas as pd


def default_loader(path):
    img = Image.open(path).convert('RGB')
    img.resize((224, 224), Image.ANTIALIAS)

    return img


class SeqDataset(Dataset):
    def __init__(self, txt, xlsx, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        rawdata = pd.read_excel(xlsx)
        rawdata=rawdata.values.tolist()
        imgseqs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            imgseqs.append(line)

        self.num_samples = len(imgseqs)
        self.imgseqs = imgseqs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.data=rawdata


    def __getitem__(self, index):
        # imagepath
        root ='imagepath'
        # current_index = np.random.choice(range(0, self.num_samples))
        imgs_path = self.imgseqs[index].split()
        current_imgs_path = imgs_path[:len(imgs_path) - 1]
        current_imgs = []
        for frame in current_imgs_path:
            img = self.loader(root + frame)
            if self.transform is not None:
                img = self.transform(img)
                current_imgs.append(img)

        data=self.data[index]
        inseq=[]
        outseq=[]
        for j in range(0,len(data)-1):
                inseq.append(data[j])
        outseq.append(data[-1])
        in_seq = torch.FloatTensor(inseq).view(-1)
        out_seq = torch.FloatTensor(outseq).view(-1)

        batch_cur_imgs = np.stack(current_imgs, axis=0)
        return batch_cur_imgs, in_seq, out_seq

    def __len__(self):
        return len(self.imgseqs)
if __name__ == '__main__':
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((128, 128))
    ]
    ##加载数据
    data_transforms = transforms.Compose(transform_list)
    train_data = SeqDataset(txt='valid_3path.txt', xlsx='valid_3in.xlsx', transform=data_transforms)
    batch_size = 20
    train_loader = DataLoader(train_data, shuffle=True, num_workers=1, batch_size=batch_size)
    print(len(train_loader))
    for i,(inputVar1, inputVar2, targetVar) in enumerate(train_loader):
        print(i)
        print(inputVar1.shape)
        print(inputVar2)
        print(targetVar)
