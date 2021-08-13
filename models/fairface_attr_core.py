import os
import pickle
import h5py
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from models import basenet
from models import dataloader
import utils
import pandas as pd
from matplotlib import image
from sklearn.metrics import average_precision_score

class FairFaceModel_attr():
    def __init__(self, opt):
        super(FairFaceModel_attr, self).__init__()
        self.epoch = 0
        self.experiment = opt['experiment']
        self.device = opt['device']
        self.save_path = opt['save_folder']
        self.print_freq = opt['print_freq']
        self.init_lr = opt['optimizer_setting']['lr']
        self.opt = opt
        self.log_writer = SummaryWriter(os.path.join(self.save_path, 'logfile'))
        
        self.set_network(opt)
        self.set_data(opt)
        self.set_optimizer(opt)
        # self.best_dev_mAP = 0.
        # self.best_train_loss = 10000000000
        self.best_train_mAP = 0.
        self.best_test_mAP = 0.

        self.test_gender = []
        self.test_race = []

    def set_network(self, opt):
        """Define the network"""
        
        self.network = basenet.ResNet50(n_classes=opt['output_dim'],
                                        pretrained=True,
                                        dropout=opt['dropout']).to(self.device)
        # self.network = basenet.Vgg16(n_classes=opt['output_dim'],
        #                                 pretrained=True,
        #                                 dropout=opt['dropout']).to(self.device)
        
    def forward(self, x):
        out, feature = self.network(x)
        return out, feature
    
    def set_data(self, opt):
        """Set up the dataloaders"""
        # normalize according to ImageNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        transform_train = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        dir = './data'
        train_imgs_df = pd.read_csv(os.path.join(dir, 'fairface_label_train.csv'))
        test_imgs_df = pd.read_csv(os.path.join(dir, 'fairface_label_val.csv'))
        
        train_image_feature = h5py.File(os.path.join(dir, 'fairface_train.h5py'), 'r')
        test_image_feature = h5py.File(os.path.join(dir, 'fairface_test.h5py'), 'r')
        
        self.train_loader = torch.utils.data.DataLoader(
            dataloader.FairFaceDataset_attr(train_image_feature, train_imgs_df, select = opt['select'], percentage = opt['percentage'], race_list = opt['race_list'], l = opt['train_size'], transform = transform_train), 
            batch_size=opt['batch_size'], shuffle=True, num_workers=1)
        if opt['experiment'].endswith('data'):
            self.test_loader = torch.utils.data.DataLoader(
                dataloader.FairFaceDataset_attr(test_image_feature, test_imgs_df, select = opt['select'], percentage = opt['percentage'], race_list = opt['race_list'], l = opt['test_size'], transform = transform_test), 
                batch_size=opt['batch_size'], shuffle=False, num_workers=1)
        elif opt['experiment'].endswith('model'):
            # keep balance in testing data, gender: 1/2, race: 1/7 or 1/4
            if opt['select'] == 'gender':
                percentage = 0.5
            else:
                percentage = 1/4
            self.test_loader = torch.utils.data.DataLoader(
                dataloader.FairFaceDataset_attr(test_image_feature, test_imgs_df, select = opt['select'], percentage = percentage, race_list = opt['race_list'], l = opt['test_size'], transform = transform_test), 
                batch_size=opt['batch_size'], shuffle=False, num_workers=1)

        # self.train_target = utils.normalized(np.array([i for i in range(opt['train_size'])]))
        
    def set_optimizer(self, opt):
        optimizer_setting = opt['optimizer_setting']
        self.optimizer = optimizer_setting['optimizer']( 
                            params=filter(lambda p: p.requires_grad, self.network.parameters()), 
                            lr=optimizer_setting['lr'],
                            weight_decay=optimizer_setting['weight_decay']
                            )
        
    def _criterion(self, output, target):
        # return F.binary_cross_entropy_with_logits(output, target[:, :-1])
        # print(output.shape)
        # print(target.shape)
        # print(output)
        # print(target)
        return F.binary_cross_entropy_with_logits(output, target)
        
    def state_dict(self):
        state_dict = {
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        return state_dict

    def log_result(self, name, result, step):
        self.log_writer.add_scalars(name, result, step)

    def _train(self, loader):
        """Train the model for one epoch"""
        
        self.network.train()
        
        train_loss = 0
        # train_targets_list = []
        for i, (images, targets, _, _) in enumerate(loader):
            # print(i)
            # print(type(images))
            # print(images.shape)
            # print(targets.shape)
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs, _ = self.forward(images)
            loss = self._criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            # train_targets_list.append(targets)
            self.log_result('Train iteration', {'loss': loss.item()},
                            len(loader)*self.epoch + i)

            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {}: [{}|{}], loss:{}'.format(
                      self.epoch+1, i+1, len(loader), loss.item()))
        
        # self.train_target = torch.cat(train_targets_list, dim = 1)
        self.log_result('Train epoch', {'loss': train_loss/len(loader)}, self.epoch+1)
        self.epoch += 1

    def _test(self, loader):
        """Compute model output on test set"""
        
        self.network.eval()

        test_loss = 0
        test_targets_list = []
        output_list = []
        feature_list = []
        gender_list = []
        race_list = []
        with torch.no_grad():
            for i, (images, targets, gender, race) in enumerate(loader):
                images, targets = images.to(self.device), targets.to(self.device)
                outputs, features = self.forward(images)
                loss = self._criterion(outputs, targets)
                test_loss += loss.item()

                test_targets_list.append(targets)
                output_list.append(outputs)
                feature_list.append(features)
                gender_list.extend(list(gender))
                race_list.extend(list(race))

        # self.test_target = torch.cat(test_targets_list, dim = 1)
        self.test_gender = gender_list
        self.test_race = race_list
        return test_loss, torch.cat(test_targets_list), torch.cat(output_list), torch.cat(feature_list)

    def inference(self, output):
        predict_prob = torch.sigmoid(output)
        return predict_prob.cpu().numpy()
    
    def train(self):
        """Train the model for one epoch, evaluate on validation set and 
        save the best model
        """
        mode = self.experiment.split("_")[-1]
        percentage = self.opt['percentage']        

        start_time = datetime.now()
        self._train(self.train_loader)
        utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, f'ckpt_{mode}_{percentage}.pth'))

        train_loss, train_target, train_output, train_feature = self._test(self.train_loader)
        train_predict_prob = self.inference(train_output)
        train_mAP = average_precision_score(train_target.cpu(), train_predict_prob)

        duration = datetime.now() - start_time
        print('Finish training epoch {}, train mAP: {}, time used: {}'.format(self.epoch, train_mAP, duration))
        self.log_result('Train epoch', {'loss': train_loss/len(self.train_loader), 'mAP': train_mAP},
                        self.epoch)
        # if train_mAP > self.best_train_mAP:
        #     self.best_train_mAP = train_mAP
        #     utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, f'best_{mode}_{percentage}.pth'))
        
        ####### Validation #######
        test_loss, test_target, test_output, test_feature = self._test(self.test_loader)
        test_predict_prob = self.inference(test_output)
        test_mAP = average_precision_score(test_target.cpu(), test_predict_prob)

        print('Testing, test mAP: {}'.format(test_mAP))
        info = ('Test loss: {}\n'
                'Test mAP: {}'.format(test_loss, test_mAP))
        if test_mAP > self.best_test_mAP:
            self.best_test_mAP = test_mAP
            utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, f'best_{mode}_{percentage}.pth'))
        
        

    def test(self):
        mode = self.experiment.split("_")[-1]
        percentage = self.opt['percentage']
        # Test and save the result
        state_dict = torch.load(os.path.join(self.save_path, f'best_{mode}_{percentage}.pth'))
        self.network.load_state_dict(state_dict['model'])
        test_loss, test_target, test_output, test_feature = self._test(self.test_loader) 
        test_predict_prob = self.inference(test_output)
        test_mAP = average_precision_score(test_target.cpu(), test_predict_prob)

        test_result = {'output': test_output.cpu().numpy(), 
                      'feature': test_feature.cpu().numpy(),
                      'gender': self.test_gender,
                      'race': self.test_race}
        utils.save_pkl(test_result, os.path.join(self.save_path, f"feature_{mode}_{percentage}.pkl"))

        print('Testing, test mAP: {}'.format(test_mAP))
        info = ('Test loss: {}\n'
                'Test mAP: {}'.format(test_loss, test_mAP))
        utils.write_info(os.path.join(self.save_path, 'result.txt'), info)

        


            
