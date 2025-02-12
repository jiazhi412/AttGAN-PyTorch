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

import torch.utils.data as data

import torch
import torchvision.utils as vutils
from models.attgan import AttGAN
from dataloader.CelebA_origin import check_attribute_conflict
from helpers import Progressbar, add_scalar_dict
from tensorboardX import SummaryWriter
import wandb


attrs_default = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]

def parse(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ'], default='CelebA')
    parser.add_argument('--data_path', dest='data_path', type=str, default='/nas/vista-ssd01/users/jiazli/datasets/CelebA/raw_data/img_align_celeba')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='/nas/vista-ssd01/users/jiazli/datasets/CelebA/raw_data/list_attr_celeba.txt')
    # parser.add_argument('--image_list_path', dest='image_list_path', type=str, default='../../datasets/CelebA/raw_image_list.txt')
    
    parser.add_argument('--img_size', dest='img_size', type=int, default=224)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=1)
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
    parser.add_argument('--gr', dest='gr', type=float, default=100.0) #
    parser.add_argument('--gc', dest='gc', type=float, default=10.0) #
    parser.add_argument('--dc', dest='dc', type=float, default=1.0) #
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0) #
    parser.add_argument('--dim_per_attr', type=int, default=1) #
    
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=4)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
    
    parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')
    
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--no_gpu', dest='gpu', action='store_false')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    
    return parser.parse_args(args)

args = parse()
print(args)

hyperparameter = f'shortcut{args.shortcut_layers}-inject{args.inject_layers}-gr{args.gr}-gc{args.gc}-dc{args.dc}-gp{args.lambda_gp}-dpa{args.dim_per_attr}'

wandb.init(project="AttGAN",
            entity="jiazhi", 
            config=args, 
            group=args.experiment_name, 
            # job_type=args.experiment_name, 
            name=hyperparameter
)

args.lr_base = args.lr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

os.makedirs(join('result', args.experiment_name), exist_ok=True)
# os.makedirs(join('result', args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join('/nas/vista-ssd01/users/jiazli/attGAN', args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join('result', args.experiment_name, 'sample_training'), exist_ok=True)
with open(join('result', args.experiment_name, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))
with open(join('/nas/vista-ssd01/users/jiazli/attGAN', args.experiment_name, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

if args.data == 'CelebA':
    from dataloader.CelebA_ import CelebA
    train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'all', args.attrs)
    valid_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs)
if args.data == 'CelebA-HQ':
    from dataloader.CelebA_origin import CelebA_HQ
    train_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'train', args.attrs)
    valid_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'valid', args.attrs)
train_dataloader = data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    shuffle=True, drop_last=True
)
valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

attgan = AttGAN(args)
progressbar = Progressbar()
# writer = SummaryWriter(join('result', args.experiment_name, 'summary'))

fixed_img_a, fixed_att_a = next(iter(valid_dataloader))
# print(fixed_img_a.max())
# print(fixed_img_a.min())
# print(fixed_img_a.size())
fixed_att_a = torch.unsqueeze(fixed_att_a,1) if len(list(fixed_att_a.size())) == 1 else fixed_att_a
# print(fixed_att_a.size())
# print(fixed_att_a)
fixed_img_a = fixed_img_a.cuda() if args.gpu else fixed_img_a
fixed_att_a = fixed_att_a.cuda() if args.gpu else fixed_att_a
fixed_att_a = fixed_att_a.type(torch.float)
sample_att_b_list = [fixed_att_a]
for i in range(args.n_attrs):
    tmp = fixed_att_a.clone()
    tmp[:, i] = 1 - tmp[:, i]
    tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
    sample_att_b_list.append(tmp)

it = 0
it_per_epoch = len(train_dataset) // args.batch_size
for epoch in range(args.epochs):
    # train with base lr in the first 100 epochs
    # and half the lr in the last 100 epochs
    lr = args.lr_base / (10 ** (epoch // 100))
    attgan.set_lr(lr)
    # writer.add_scalar('LR/learning_rate', lr, it+1)
    for img_a, att_a in progressbar(train_dataloader):
        attgan.train()
        
        att_a = torch.unsqueeze(att_a,1) if len(list(att_a.size())) == 1 else att_a
        img_a = img_a.cuda() if args.gpu else img_a
        att_a = att_a.cuda() if args.gpu else att_a
        idx = torch.randperm(len(att_a))
        att_b = att_a[idx].contiguous()
        
        att_a = att_a.type(torch.float)
        att_b = att_b.type(torch.float)
        
        att_a_ = (att_a * 2 - 1) * args.thres_int # -1/2, 1/2 for all
        # print(att_a)
        # print(att_a_)
        if args.b_distribution == 'none':
            att_b_ = (att_b * 2 - 1) * args.thres_int
        if args.b_distribution == 'uniform':
            att_b_ = (att_b * 2 - 1) * \
                     torch.rand_like(att_b) * \
                     (2 * args.thres_int)
        if args.b_distribution == 'truncated_normal':
            att_b_ = (att_b * 2 - 1) * \
                     (torch.fmod(torch.randn_like(att_b), 2) + 2) / 4.0 * \
                     (2 * args.thres_int)
        
        if (it+1) % (args.n_d+1) != 0:
            # print('dlkjasdj')
            errD = attgan.trainD(img_a, att_a, att_a_, att_b, att_b_)
            # add_scalar_dict(writer, errD, it+1, 'D')
        else:
            errG = attgan.trainG(img_a, att_a, att_a_, att_b, att_b_)
            # add_scalar_dict(writer, errG, it+1, 'G')
            progressbar.say(epoch=epoch, iter=it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])
            # progressbar.say(epoch=epoch, iter=it+1, g_loss=errG['g_loss'])
        
        if (it+1) % args.save_interval == 0:
            # To save storage space, I only checkpoint the weights of G.
            # If you'd like to keep weights of G, D, optim_G, optim_D,
            # please use save() instead of saveG().
            attgan.saveG(os.path.join(
                'result', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            ))
            # attgan.save(os.path.join(
            #     'result', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            # ))
        if (it+1) % args.sample_interval == 0:
            attgan.eval()
            with torch.no_grad():
                samples = [fixed_img_a]
                for i, att_b in enumerate(sample_att_b_list):
                    att_b_ = (att_b * 2 - 1) * args.thres_int # -1/2, 1/2 for all
                    if i > 0: # i == 0 is for reconstruction
                        att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int # -1, 1 for interested att; -1/2, 1/2 for others
                    samples.append(attgan.G(fixed_img_a, att_b_))
                samples.append((samples[-1] + samples[-2])/2)
                samples.append(attgan.G(fixed_img_a, torch.zeros_like(att_b_)))
                samples = torch.cat(samples, dim=3)
                # writer.add_image('sample', vutils.make_grid(samples, nrow=1, normalize=True, range=(-1., 1.)), it+1)
                vutils.save_image(samples, os.path.join(
                        'result', args.experiment_name, 'sample_training',
                        'Epoch_({:d})_({:d}of{:d}).jpg'.format(epoch, it%it_per_epoch+1, it_per_epoch)
                    ), nrow=1, normalize=False, range=(0., 1.))
                wandb.log({'test/filtered images': wandb.Image(vutils.make_grid(samples, nrow=1, padding=0, normalize=False))})
        it += 1