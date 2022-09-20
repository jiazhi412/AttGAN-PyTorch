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

torch.cuda.empty_cache()

attrs_default = [
    'Male', 
]

def parse(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ'], default='CelebA')
    parser.add_argument('--data_path', dest='data_path', type=str, default='/nas/vista-ssd01/users/jiazli/datasets/CelebA/raw_data/img_align_celeba')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='/nas/vista-ssd01/users/jiazli/datasets/CelebA/raw_data/list_attr_celeba.txt')
    
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
    parser.add_argument('--dc', dest='dc', type=float, default=1.0) #
    parser.add_argument('--gp', dest='gp', type=float, default=10.0) # default in WGAN-GP
    parser.add_argument('--ga', dest='ga', type=float, default=5.0) # TODO
    # parser.add_argument('--gf', dest='gf', type=float, default=1.0) #
    parser.add_argument('--dim_per_attr', type=int, default=1) # TODO
    parser.add_argument('--num_ganp1', type=int, default=1) # TODO
    parser.add_argument('--num_ganp2', type=int, default=1) # TODO 
    parser.add_argument('--num_dis', type=int, default=1) # TODO
    
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan']) #
    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=4)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='learning rate')
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
args.hyperparameter = f'shortcut{args.shortcut_layers}-inject{args.inject_layers}-gr{args.gr}-gc{args.gc}-dc{args.dc}-gp{args.gp}-ga{args.ga}-dpa{args.dim_per_attr}'

wandb.init(project="AttGAN",
            entity="jiazhi", 
            config=args, 
            group=args.experiment, 
            job_type=args.name, 
            name=args.hyperparameter
)

os.makedirs(join('result', args.experiment, args.name, args.hyperparameter), exist_ok=True)
os.makedirs(join('result', args.experiment, args.name, args.hyperparameter, 'checkpoint'), exist_ok=True)
os.makedirs(join('result', args.experiment, args.name, args.hyperparameter, 'sample_training'), exist_ok=True)
with open(join('result', args.experiment, args.name, args.hyperparameter, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

# choose model
if args.experiment == 'label_mse':
    from models.CelebA_label_mse import Model
    attgan = Model(args)
elif args.experiment == 'label':
    from models.CelebA_label import AttGAN
    attgan = AttGAN(args)




# choose dataset
if args.data == 'CelebA':
    from dataloader.CelebA_ import CelebA
    train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'all', args.attrs)
    valid_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs)
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
    attgan.train_epoch(train_dataloader, valid_dataloader, it_per_epoch, args)