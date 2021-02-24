import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from matplotlib import image

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
    def __init__(self, imgs, df):
        self.imgs = imgs
        self.labels = [i for i in range(len(df))]
        self.df = df
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        gender = self.pd['gender'].tolist()[idx]
        race = self.pd['race'].tolist()[idx]
        return color_img, label, gender, race