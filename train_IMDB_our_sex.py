# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Main entry point for training AttGAN network."""

import argparse
import datetime
import json
import os
from os.path import join
import torch
import torch.utils.data as data
import wandb
import utils
import h5py
import torchvision.transforms as transforms

torch.cuda.empty_cache()

attrs_default = [
    'Male', 
]

def parse(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to eliminate')
    parser.add_argument('--attrs_d', default=['Blond_Hair'], nargs='+', help='attributes for downstream')
    parser.add_argument('--data', dest='data', type=str, default='IMDB')
    parser.add_argument('--data_path', dest='data_path', type=str, default='/nas/vista-ssd01/users/jiazli/datasets/CelebA/raw_data/img_align_celeba')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='/nas/vista-ssd01/users/jiazli/datasets/CelebA/raw_data/list_attr_celeba.txt')
    parser.add_argument('--syn_data_path', dest='syn_data_path', type=str, default='/nas/vista-ssd01/users/jiazli/synthetic_datasets/CelebA/processed_data/blond_hair')
    
    parser.add_argument('--img_size', dest='img_size', type=int, default=224)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=0) #
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0) #
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--gr', dest='gr', type=float, default=100.0) # TODO
    parser.add_argument('--gc', dest='gc', type=float, default=10.0) # TODO
    parser.add_argument('--dc', dest='dc', type=float, default=1.0) # TODO
    parser.add_argument('--gp', dest='gp', type=float, default=10.0) # default in WGAN-GP
    parser.add_argument('--ga', dest='ga', type=float, default=5.0) # TODO
    parser.add_argument('--mi', dest='mi', type=float, default=5.0) # TODO
    parser.add_argument('--num_iter_MI', type=int, default=40) # TODO
    # parser.add_argument('--gf', dest='gf', type=float, default=1.0) #
    parser.add_argument('--dim_per_attr', type=int, default=1) # TODO
    parser.add_argument('--num_ganp1', type=int, default=1) # TODO
    parser.add_argument('--num_ganp2', type=int, default=1) # TODO 
    parser.add_argument('--num_dis', type=int, default=1) # TODO

    parser.add_argument("--IMDB_train_mode", type=str, choices=['eb1', 'eb2', 'eb1_ex', 'eb2_ex', 'all'], default='all')
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan']) #
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=4)
    parser.add_argument('--lr', dest='lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0) # default in WGAN-GP
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.9) # default in WGAN-GP
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
    
    parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')
    
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--no_gpu', dest='gpu', action='store_false')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument("--experiment", metavar="",)
    parser.add_argument('--name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    
    return parser.parse_args(args)

args = parse()
print(args)
args.lr_base = args.lr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)
# args.hyperparameter = f'shortcut{args.shortcut_layers}-inject{args.inject_layers}-gr{args.gr}-gc{args.gc}-dc{args.dc}-gp{args.gp}-ga{args.ga}-mi{args.mi}'
args.hyperparameter = f'gr{args.gr}-gc{args.gc}-dc{args.dc}-ga{args.ga}-mi{args.mi}-{args.num_ganp1}-{args.num_ganp2}-{args.num_dis}'

wandb.init(project="AttGAN",
            entity="jiazhi", 
            config=args, 
            group=args.experiment, 
            job_type=args.name, 
            name=args.hyperparameter
)

os.makedirs(join('result', args.experiment, args.name, args.IMDB_train_mode, args.hyperparameter), exist_ok=True)
# os.makedirs(join('result', args.experiment, args.name, args.IMDB_train_mode, args.hyperparameter, 'checkpoint'), exist_ok=True)
os.makedirs(join('/nas/vista-ssd01/users/jiazli/attGAN', args.experiment, args.name, args.IMDB_train_mode, args.hyperparameter, 'checkpoint'), exist_ok=True)
os.makedirs(join('result', args.experiment, args.name, args.IMDB_train_mode, args.hyperparameter, 'sample_training'), exist_ok=True)
with open(join('result', args.experiment, args.name, args.IMDB_train_mode, args.hyperparameter, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))
with open(join('/nas/vista-ssd01/users/jiazli/attGAN', args.experiment, args.name, args.IMDB_train_mode, args.hyperparameter, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

# choose model
if args.experiment == 'label':
    from models.CelebA_label import Model 
    m = Model(args)
elif args.experiment == 'label_mse':
    from models.CelebA_label_mse import Model
    m = Model(args)
elif args.experiment == 'label_mse_MI':
    from models.CelebA_label_mse_MI import Model
    m = Model(args)
elif args.experiment == 'label_mse_MI_Pscratch':
    from models.CelebA_label_mse_MI_Pscratch import Model
    m = Model(args)
elif args.experiment == 'attGAN_MI':
    from models.IMDB_attgan_MI import Model
    m = Model(args)
elif args.experiment == 'attGAN':
    from models.IMDB_attgan import Model
    m = Model(args)
elif args.experiment == 'label_mse_MI_PDshare':
    from models.IMDB_label_mse_MI_PDshare import Model
    m = Model(args)




# choose dataset
if args.data == 'IMDB':
    from dataloader.IMDB import IMDBDataset 
    data_folder= {
        'origin_image_feature_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/IMDB.h5py',
        'origin_target_dict_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/age_dict',
        'origin_sex_dict_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/sex_dict',
        'origin_eb1_key_list_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/eb1_img_list',
        'origin_eb2_key_list_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/eb2_img_list',
        'origin_unbiased_key_list_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/test_img_list',
        'augment': False
    }
    # image_feature = h5py.File(data_folder['origin_image_feature_path'], 'r')
    image_feature_path = data_folder['origin_image_feature_path']
    target_dict = utils.load_pkl(data_folder['origin_target_dict_path'])
    sex_dict = utils.load_pkl(data_folder['origin_sex_dict_path'])
    eb1_key_list = utils.load_pkl(data_folder['origin_eb1_key_list_path'])
    eb2_key_list = utils.load_pkl(data_folder['origin_eb2_key_list_path'])
    unbiased_key_list = utils.load_pkl(data_folder['origin_unbiased_key_list_path'])
    
    transform = transforms.Compose([
                                    transforms.Resize((args.img_size, args.img_size)),
                                    transforms.CenterCrop(180),
                                    transforms.Resize((args.img_size, args.img_size)),
                                    transforms.ToTensor(),
                                    # normalize,
                                    ])
    train_dataset = IMDBDataset(image_feature_path, target_dict, sex_dict, args.IMDB_train_mode, eb1_key_list, eb2_key_list, unbiased_key_list, 'train', transform)
    valid_dataset = IMDBDataset(image_feature_path, target_dict, sex_dict, 'all', eb1_key_list, eb2_key_list, unbiased_key_list, 'dev', transform)



train_dataloader = data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    shuffle=True, drop_last=True
)
valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))




# start epoch
it_per_epoch = len(train_dataset) // args.batch_size
for epoch in range(args.epochs):
    m.train_epoch(train_dataloader, valid_dataloader, it_per_epoch, args)