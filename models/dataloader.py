import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from matplotlib import image
from PIL import Image
import utils
import numpy as np

def load_data(dir, opt):
    train_imgs_df = pd.read_csv(os.path.join(dir, 'fairface_label_train.csv'))
    test_imgs_df = pd.read_csv(os.path.join(dir, 'fairface_label_val.csv'))

    train_imgs_list = []
    for index, img_path in enumerate(train_imgs_df['file']):
        img = image.imread(os.path.join(dir, img_path))
        train_imgs_list.append(img)
    train_loader = DataLoader(
        FairFaceDataset(train_imgs_list, train_imgs_df), 
        batch_size=opt['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    test_imgs_list = []
    for index, img_path in enumerate(test_imgs_df['file']):
        img = image.imread(os.path.join(dir, img_path))
        test_imgs_list.append(img)
    test_loader = DataLoader(
        FairFaceDataset(test_imgs_list, test_imgs_df), 
        batch_size=len(test_imgs_list), shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


# def load(imgs_df):
#     imgs_list = []
#     for index, img_path in enumerate(imgs_df['file']):
#         img = image.open(img_path)
#         imgs_list.append(img)
#     loader = DataLoader(
#         FairFaceDataset(imgs_list, imgs_df), 
#         batch_size=opt['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
#     return loader


class FairFaceDataset(Dataset):
    def __init__(self, imgs, df, female_percentage, select_flag, l=None, transform=None):
        self.imgs = imgs
        # fp = 0.9
        if select_flag:
            self.df = select(df, female_percentage, l)
        else:
            self.df = df
        self.transform = transform
        # if l != None:
        #     self.l = l
        # else:
        #     self.l = len(self.df)
        self.l = len(self.df)
        self.labels = [i for i in range(self.l)]
        self.labels = utils.normalized(np.array(self.labels))
    
    def __len__(self):
        return self.l
    
    def __getitem__(self, idx):
        key = self.df.loc[idx, 'file']
        img = self.imgs[key][()]
        label = self.labels[idx]
        gender = self.df.loc[idx, 'gender']
        race = self.df.loc[idx, 'race']

        if self.transform is not None:
            img = self.transform(img)
        # print(label)
        # print(torch.FloatTensor(label))
        # return img, torch.FloatTensor(label)
        return img, label, gender, race


def select(df, female_percentage, l):
    female = df[df["gender"] == "Female"]
    male = df[df["gender"] == "Male"]
    res = pd.concat([female.head(int(l*female_percentage)), male.head(int(l- l * female_percentage))])
    res.index = range(len(res))
    return res
