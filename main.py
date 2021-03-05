import torch
import pandas as pd
import os
# from utils.datasets import *
from matplotlib import image
from parse_args import *


# opt = collect_args()

# # load data
# dir = './data'
# train_loader, test_loader = load_data(dir, opt)

# a = 10
# # train
# # ResNet
# # test


import utils

def main(model, opt):
    utils.set_random_seed(opt['random_seed'])
    
    if opt['mode'] == 'train':
        for epoch in range(opt['total_epochs']):
            model.train()
    # print('1')
    model.test()

if __name__ == '__main__':
    model, opt = collect_args()
    main(model, opt)
