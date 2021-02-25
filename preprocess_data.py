import pandas as pd
import utils
import h5py
from PIL import Image
import os
import numpy as np 

dir = './data'


train_imgs_df = pd.read_csv(os.path.join(dir, 'fairface_label_train.csv'))
test_imgs_df = pd.read_csv(os.path.join(dir, 'fairface_label_val.csv'))

# images
# train_imgs_list = []
# for index, img_path in enumerate(train_imgs_df['file']):
#     img = image.imread(os.path.join(dir, img_path))
#     train_imgs_list.append(img)
# utils.save_pkl(train_imgs_list, 'train_imgs_list.pkl')

# test_imgs_list = []
# for index, img_path in enumerate(train_imgs_df['file']):
#     img = np.asarray(Image.open(os.path.join(dir, img_path)).convert('RGB'))
#     test_imgs_list.append(img)
# utils.save_pkl(test_imgs_list, 'test_imgs_list.pkl')


# images
train_feature_file = h5py.File(os.path.join(dir, 'fairface_train.h5py'), "w")
for index, img_path in enumerate(train_imgs_df['file']):
    train_feature_file.create_dataset(img_path, 
        data=np.asarray(np.asarray(Image.open(os.path.join(dir, img_path)).convert('RGB'))))
train_feature_file.close()

test_feature_file = h5py.File(os.path.join(dir, 'fairface_test.h5py'), "w")
for index, img_path in enumerate(test_imgs_df['file']):
    test_feature_file.create_dataset(img_path, 
        data=np.asarray(np.asarray(Image.open(os.path.join(dir, img_path)).convert('RGB'))))
test_feature_file.close()

# test = h5py.File(os.path.join(dir, 'test.h5py'), "w")
# test.create_dataset('1', data = np.asarray(Image.open(os.path.join(dir, 'train/1.jpg')).convert('RGB')))
# test_feature_file.close()

# gender and race
train_gender = train_imgs_df['gender'].tolist()
train_race = train_imgs_df['race'].tolist()
test_gender = test_imgs_df['gender'].tolist()
test_race = test_imgs_df['race'].tolist()

utils.save_pkl(train_gender, os.path.join(dir, 'train_gender.pkl'))
utils.save_pkl(train_race, os.path.join(dir, 'train_race.pkl'))
utils.save_pkl(test_gender, os.path.join(dir, 'test_gender.pkl'))
utils.save_pkl(test_race, os.path.join(dir, 'test_race.pkl'))