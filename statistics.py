import pandas as pd
import os 

dir = './data'
train_imgs_df = pd.read_csv(os.path.join(dir, 'fairface_label_train.csv'))
test_imgs_df = pd.read_csv(os.path.join(dir, 'fairface_label_val.csv'))

Black = train_imgs_df[train_imgs_df['race'] == 'Black']
White = train_imgs_df[train_imgs_df['race'] == 'White']
Indian = train_imgs_df[train_imgs_df['race'] == 'Indian']
EAsian = train_imgs_df[train_imgs_df['race'] == 'East Asian']
print(train_imgs_df.groupby('race').count())