import pandas as pd
import utils
import h5py
from PIL import Image

dir = './data'

train_feature_file = h5py.File(os.path.join(dir, 'fairface_train.h5py'), "w")
for filename in os.listdir(os.path.join(dir, 'train')):
    feature_file.create_dataset(filename, 
    data=np.asarray(Image.open(os.path.join(dir, 'train', filename)).convert('RGB')))
feature_file.close()

train_imgs_df = pd.read_csv(os.path.join(opt['dir'], 'fairface_label_train.csv'))
test_imgs_df = pd.read_csv(os.path.join(opt['dir'], 'fairface_label_val.csv'))




train_gender = train_imgs_df['gender'].tolist()
train_race = train_imgs_df['race'].tolist()

test_gender = test_imgs_df['gender'].tolist()
test_race = test_imgs_df['race'].tolist()

utils.save_pkl(train, )