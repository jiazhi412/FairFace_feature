import argparse
import torch
import os
import models
import utils

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',
                        choices=[
                                 'fairface_baseline',
                                ], type=str, default='fairface_baseline')

    parser.add_argument('--experiment-name', type=str, default='debug', help='specifies a name to this experiment for saving the model and result)')
    # parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--with-cuda', dest='cuda', action='store_true')
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--mode', choices=['train', 'test'], type=str, default='train')

    parser.add_argument('--train-size', default=50000, type=int)
    parser.add_argument('--test-size', default=10000, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    args = parser.parse_args()

    opt = vars(parser.parse_args())
    opt['dir'] = './data'

    model, opt = create_experiment_setting(opt)
    return model, opt

def create_experiment_setting(opt):
    # common experiment setting
    if opt['experiment'].startswith('fairface'):
        opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
        opt['print_freq'] = 50
        # opt['batch_size'] = 32
        opt['total_epochs'] = 50
        opt['save_folder'] = os.path.join('record', opt['experiment'], opt['experiment_name'])
        utils.creat_folder(opt['save_folder'])
        opt['output_dim'] = 1

        optimizer_setting = {
            'optimizer': torch.optim.Adam,
            'lr': 1e-4,
            'weight_decay': 0,
        }
        opt['optimizer_setting'] = optimizer_setting
        opt['dropout'] = 0.5


    if opt['experiment'] == 'fairface_baseline':
        model = models.fairface_core.FairFaceModel(opt)



    return model, opt
