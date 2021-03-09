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

class FairFaceModel():
    def __init__(self, opt):
        super(FairFaceModel, self).__init__()
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
        self.best_dev_mAP = 0.
        self.best_train_loss = 10000000000

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
            dataloader.FairFaceDataset(train_image_feature, train_imgs_df, select = opt['select'], percentage = opt['percentage'], l = opt['train_size'], transform = transform_train), 
            batch_size=opt['batch_size'], shuffle=True, num_workers=1)
        if opt['experiment'].endswith('data'):
            self.test_loader = torch.utils.data.DataLoader(
                dataloader.FairFaceDataset(test_image_feature, test_imgs_df, select = opt['select'], percentage = opt['percentage'], l = opt['test_size'], transform = transform_test), 
                batch_size=opt['batch_size'], shuffle=False, num_workers=1)
        elif opt['experiment'].endswith('model'):
            self.test_loader = torch.utils.data.DataLoader(
                dataloader.FairFaceDataset(test_image_feature, test_imgs_df, select = opt['select'], percentage = 0.5, l = opt['test_size'], transform = transform_test), 
                batch_size=opt['batch_size'], shuffle=False, num_workers=1)

        self.train_target = utils.normalized(np.array([i for i in range(opt['train_size'])]))
        
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
        return F.binary_cross_entropy_with_logits(output, target.view((-1, 1)))
        
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
            self.log_result('Train iteration', {'loss': loss.item()},
                            len(loader)*self.epoch + i)

            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {}: [{}|{}], loss:{}'.format(
                      self.epoch, i+1, len(loader), loss.item()))
        
        self.log_result('Train epoch', {'loss': train_loss/len(loader)}, self.epoch)
        self.epoch += 1

    def _test(self, loader):
        """Compute model output on test set"""
        
        self.network.eval()

        test_loss = 0
        output_list = []
        feature_list = []
        gender_list = []
        race_list = []
        with torch.no_grad():
            for i, (images, targets, gender, race) in enumerate(loader):
                # print("2.5")
                images, targets = images.to(self.device), targets.to(self.device)
                outputs, features = self.forward(images)
                loss = self._criterion(outputs, targets)
                test_loss += loss.item()
                # print('2.6')
                output_list.append(outputs)
                feature_list.append(features)
                gender_list.extend(list(gender))
                race_list.extend(list(race))

                # print(gender)
                # print(gender_list)
        self.test_gender = gender_list
        self.test_race = race_list
        return test_loss, torch.cat(output_list), torch.cat(feature_list)

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

        train_loss, train_output, train_feature = self._test(self.train_loader)
        train_predict_prob = self.inference(train_output)
        self.log_result('Train epoch', {'loss': train_loss/len(self.train_loader)}, self.epoch)
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, f'best_{mode}_{percentage}.pth'))
        
        # train_result = {'output': train_output.cpu().numpy(), 
        #               'feature': train_feature.cpu().numpy(),
        #               'gender': self.train_gender,
        #               'race': self.train_race}
        # utils.save_pkl(train_result, os.path.join(self.save_path, 'train_result.pkl'))
        
        duration = datetime.now() - start_time
        print('Finish training epoch {}, time used: {}'.format(self.epoch, duration))


        # train_mAP = average_precision_score(self.train_target, train_predict_prob)

        # train_per_class_AP = utils.compute_weighted_AP(self.train_target, train_predict_prob, 
        #                                              self.train_class_weight)
        # train_mAP = utils.compute_mAP(train_per_class_AP, self.subclass_idx)
        
        # self.log_result('Train epoch', {'loss': train_loss/len(self.train_loader), 'mAP': train_mAP},
        #                 self.epoch)
        # if train_mAP > self.best_train_mAP:
        #     self.best_train_mAP = train_mAP
        #     utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'best.pth'))
        
        # duration = datetime.now() - start_time
        # print('Finish training epoch {}, train mAP: {}, time used: {}'.format(self.epoch, train_mAP, duration))

        # dev_loss, dev_output, _ = self._test(self.dev_loader)
        # dev_predict_prob = self.inference(dev_output)
        # dev_per_class_AP = utils.compute_weighted_AP(self.dev_target, dev_predict_prob, 
        #                                              self.dev_class_weight)
        # dev_mAP = utils.compute_mAP(dev_per_class_AP, self.subclass_idx)
        
        # self.log_result('Dev epoch', {'loss': dev_loss/len(self.dev_loader), 'mAP': dev_mAP},
        #                 self.epoch)
        # if dev_mAP > self.best_dev_mAP:
        #     self.best_dev_mAP = dev_mAP
        #     utils.save_state_dict(self.state_dict(), os.path.join(self.save_path, 'best.pth'))
        
        # duration = datetime.now() - start_time
        # print('Finish training epoch {}, dev mAP: {}, time used: {}'.format(self.epoch, dev_mAP, duration))

    def test(self):
        mode = self.experiment.split("_")[-1]
        percentage = self.opt['percentage']
        # Test and save the result
        state_dict = torch.load(os.path.join(self.save_path, f'best_{mode}_{percentage}.pth'))
        # print('1.5')
        self.network.load_state_dict(state_dict['model'])
        # print('2')
        test_loss, test_output, test_feature = self._test(self.test_loader) 
        # print('3')
        test_result = {'output': test_output.cpu().numpy(), 
                      'feature': test_feature.cpu().numpy(),
                      'gender': self.test_gender,
                      'race': self.test_race}
        
        utils.save_pkl(test_result, os.path.join(self.save_path, f"feature_{mode}_{percentage}.pkl"))
        # Output the mean AP for the best model on dev and test set
        info = ('Test loss: {}\n'.format(test_result))
        utils.write_info(os.path.join(self.save_path, 'result.txt'), info)
        
        
        # dev_loss, dev_output, dev_feature = self._test(self.dev_loader)
        # dev_predict_prob = self.inference(dev_output)
        # dev_per_class_AP = utils.compute_weighted_AP(self.dev_target, dev_predict_prob, 
        #                                              self.dev_class_weight)
        # dev_mAP = utils.compute_mAP(dev_per_class_AP, self.subclass_idx)
        # dev_result = {'output': dev_output.cpu().numpy(), 
        #               'feature': dev_feature.cpu().numpy(),
        #               'per_class_AP': dev_per_class_AP,
        #               'mAP': dev_mAP}
        # utils.save_pkl(dev_result, os.path.join(self.save_path, 'dev_result.pkl'))
        
        # test_loss, test_output, test_feature = self._test(self.test_loader)
        # test_predict_prob = self.inference(test_output)
        # test_per_class_AP = utils.compute_weighted_AP(self.test_target, test_predict_prob, 
        #                                              self.test_class_weight)
        # test_mAP = utils.compute_mAP(test_per_class_AP, self.subclass_idx)
        # test_result = {'output': test_output.cpu().numpy(), 
        #               'feature': test_feature.cpu().numpy(),
        #               'per_class_AP': test_per_class_AP,
        #               'mAP': test_mAP}
        # utils.save_pkl(test_result, os.path.join(self.save_path, 'test_result.pkl'))
        
        # # Output the mean AP for the best model on dev and test set
        # info = ('Dev mAP: {}\n'
        #         'Test mAP: {}'.format(dev_mAP, test_mAP))
        # utils.write_info(os.path.join(self.save_path, 'result.txt'), info)
    


            
