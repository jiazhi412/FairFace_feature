import argparse
import torch
import os
import models
import utils

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',
                        choices=[
                                 'FairFace_baseline_data',
                                 'FairFace_baseline_model',
                                 'FairFace_attr_baseline_model'
                                ], type=str, default='FairFace_baseline_data')

    parser.add_argument('--experiment-name', type=str, default='debug', help='specifies a name to this experiment for saving the model and result)')
    parser.add_argument('--with-cuda', dest='cuda', action='store_true')
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--mode', choices=['train', 'test'], type=str, default='train')

    parser.add_argument('--select', default="Black", type=str)
    parser.add_argument('--percentage', default=0.9, type=float)

    parser.add_argument('--train-size', default=40000, type=int)
    parser.add_argument('--test-size', default=5000, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--epochs', dest='total_epochs', default=50, type=int)
    args = parser.parse_args()

    opt = vars(parser.parse_args())
    opt['dir'] = './data'

    model, opt = create_experiment_setting(opt)
    return model, opt

def create_experiment_setting(opt):
    # common experiment setting
    if opt['experiment'].startswith('FairFace'):
        opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
        opt['print_freq'] = 50
        opt['save_folder'] = os.path.join('record', opt['experiment'], opt['experiment_name'])
        utils.creat_folder(opt['save_folder'])
        opt['output_dim'] = 18

        optimizer_setting = {
            'optimizer': torch.optim.Adam,
            'lr': 1e-4,
            'weight_decay': 0,
        }
        opt['optimizer_setting'] = optimizer_setting
        opt['dropout'] = 0.5


    if opt['experiment'].startswith('FairFace_baseline'):
        model = models.fairface_core.FairFaceModel(opt)
    elif opt['experiment'].startswith('FairFace_attr_baseline'):
        model = models.fairface_attr_core.FairFaceModel_attr(opt)

    return model, opt