import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from matplotlib import image
from PIL import Image
import utils
import numpy as np
from sklearn import preprocessing

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
    def __init__(self, imgs, df, select, percentage, l=None, transform=None):
        self.imgs = imgs
        if select == 'gender':
            self.df = select_gender(df, percentage, l)
        elif select in ['Black', 'White', 'Indian', 'East_Asian']:
            if select == "East_Asian":
                select = "East Asian"
            self.df = select_race(df, select, percentage, l)
        else:
            self.df = df
        self.transform = transform
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

def category2onehot(category):
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(category)
    newCol = []
    for c in category:
        c_array = np.array(c).reshape(-1, 1)
        newCol.append(encoder.transform(c_array).toarray())
    onehot = np.array(newCol).reshape(category.shape[0],-1)
    onehot = torch.tensor(onehot)
    return onehot

# gender_dict = {'Female':0, 'Male':1}
# race_dict = {'East Asian': 0, 'White': 1, 'Latino_Hispanic': 2, 'Southeast Asian': 3, 'Black': 4, 'Indian': 5, 'Middle Eastern': 6}
# age_dict = {'3-9': 0, '10-19': 1, '20-29': 2, '30-39': 3, '40-49': 4, '50-59': 5, '60-69': 6, 'more than 70': 7}

class FairFaceDataset_attr(Dataset):
    def __init__(self, imgs, df, select, percentage, l=None, transform=None):
        self.imgs = imgs
        if select == 'gender':
            self.df = select_gender(df, percentage, l)
        elif select in ['Black', 'White', 'Indian', 'East_Asian']:
            if select == "East_Asian":
                select = "East Asian"
            self.df = select_race(df, select, percentage, l)
        else:
            self.df = df
        self.transform = transform
        self.l = len(self.df)
        # self.labels = [i for i in range(self.l)]
        # self.labels = utils.normalized(np.array(self.labels))

        self.gender = category2onehot(self.df['gender'].to_numpy().reshape((-1,1)))
        self.race = category2onehot(self.df['race'].to_numpy().reshape(-1,1))
        self.age = category2onehot(self.df['age'].to_numpy().reshape(-1,1)) 
        self.attrs = np.concatenate((self.gender, self.race, self.age), axis=1)
        
    
    def __len__(self):
        return self.l
    
    def __getitem__(self, idx):
        key = self.df.loc[idx, 'file']
        img = self.imgs[key][()]
        # label = self.labels[idx]
        gender = self.df.loc[idx, 'gender']
        race = self.df.loc[idx, 'race']
        age = self.df.loc[idx, 'age']

        attr = self.attrs[idx]

        if self.transform is not None:
            img = self.transform(img)
        # print(label)
        # print(torch.FloatTensor(label))
        # return img, torch.FloatTensor(label)
        return img, attr, gender, race

def select_gender(df, percentage, l):
    female = df[df["gender"] == "Female"]
    male = df[df["gender"] == "Male"]
    res = pd.concat([female.head(int(l*percentage)), male.head(int(l- l * percentage))])
    res.index = range(len(res))
    return res

def select_race(df, select, percentage, l):
    race = df[df["race"] == select]
    other = df[df["race"] != select]
    res = pd.concat([race.head(int(l*percentage)), other.head(int(l- l * percentage))])
    res.index = range(len(res))
    return res
