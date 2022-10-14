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
import torchvision.utils as vutils
import wandb
from helpers import Progressbar
from torchvision import datasets, transforms

from dataloader.CMNIST import ColoredDataset_generated


def parse(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', dest='data', type=str, default='CMNIST')
    parser.add_argument('--data_path', dest='data_path', type=str, default="/nas/vista-ssd01/users/jiazli/datasets/MNIST")
    parser.add_argument("--biased_var", type=float, default=-1) # -1 uniformly
    
    parser.add_argument('--img_size', dest='img_size', type=int, default=32)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=3)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=3)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--gr', dest='gr', type=float, default=100.0) #
    parser.add_argument('--gc', dest='gc', type=float, default=300.0) #
    parser.add_argument('--dc', dest='dc', type=float, default=1.0) #
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0) #
    parser.add_argument('--dim_per_attr', type=int, default=10) #
    
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=14, help='# of epochs')
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
    # parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    parser.add_argument("--experiment", metavar="",)
    parser.add_argument('--name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    
    return parser.parse_args(args)

args = parse()
print(args)

args.lr_base = args.lr
args.n_attrs = 3 # RGB
args.betas = (args.beta1, args.beta2)
args.hyperparameter = f'shortcut{args.shortcut_layers}-inject{args.inject_layers}-gr{args.gr}-gc{args.gc}-dc{args.dc}-gp{args.lambda_gp}-dpa{args.dim_per_attr}'

wandb.init(project="AttGAN",
            entity="jiazhi", 
            config=args, 
            group=args.experiment, 
            job_type=args.name, 
            name=args.hyperparameter
)

os.makedirs(join('result', args.experiment, args.name, str(float(args.biased_var))), exist_ok=True)
# os.makedirs(join('result', args.experiment_name, str(float(args.biased_var)), args.hyperparameter,  'checkpoint'), exist_ok=True)
os.makedirs(join('/nas/vista-ssd01/users/jiazli/attGAN', args.experiment, args.name, str(float(args.biased_var)), args.hyperparameter, 'checkpoint'), exist_ok=True)
os.makedirs(join('result', args.experiment, args.name, str(float(args.biased_var)), args.hyperparameter,  'sample_training'), exist_ok=True)
with open(join('result', args.experiment, args.name, str(float(args.biased_var)), args.hyperparameter, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))
with open(join('/nas/vista-ssd01/users/jiazli/attGAN', args.experiment, args.name, str(float(args.biased_var)), args.hyperparameter, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

if args.data == 'CMNIST':
    # load grey scale data to generate dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_set_grey = datasets.MNIST(root=args.data_path, train=True, download=False, transform=transform)
    test_set_grey = datasets.MNIST(root=args.data_path, train=False, download=False, transform=transform)
    train_set_grey, dev_set_grey = torch.utils.data.random_split(train_set_grey, [50000, 10000])

    train_dataset = ColoredDataset_generated(train_set_grey, var=args.biased_var)
    valid_dataset = ColoredDataset_generated(dev_set_grey, var=args.biased_var)
    test_set = ColoredDataset_generated(test_set_grey, var=args.biased_var)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.n_samples, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.n_samples, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.n_samples, shuffle=True, num_workers=4, pin_memory=True)


print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

if args.experiment == 'attGAN_pretrain':
    from models.CMNIST_attgan_pretrain import AttGAN
    attgan = AttGAN(args)
if args.experiment == 'attGAN':
    from models.CMNIST_attgan import AttGAN
    attgan = AttGAN(args)
if args.experiment == 'attGAN_PDsplit':
    from models.CMNIST_attgan_PDsplit import AttGAN
    attgan = AttGAN(args)
progressbar = Progressbar()

fixed_img_a, fixed_att_a = next(iter(valid_dataloader))
fixed_img_a = fixed_img_a.cuda() if args.gpu else fixed_img_a
fixed_att_a = fixed_att_a.cuda() if args.gpu else fixed_att_a
sample_att_b_list = [fixed_att_a, torch.ones_like(fixed_att_a), torch.zeros_like(fixed_att_a)]




it = 0
it_per_epoch = len(train_dataset) // args.batch_size
for epoch in range(args.epochs):
    # train with base lr in the first 100 epochs
    # and half the lr in the last 100 epochs
    lr = args.lr_base / (10 ** (epoch // 100))
    attgan.set_lr(lr)
    #writer.add_scalar('LR/learning_rate', lr, it+1)
    for img_a, att_a in progressbar(train_dataloader):
        attgan.train()
        
        # att_a = torch.unsqueeze(att_a,1) if len(list(att_a.size())) == 1 else att_a
        img_a = img_a.cuda() if args.gpu else img_a
        att_a = att_a.cuda() if args.gpu else att_a

        att_b = torch.ones_like(att_a)
        
        att_a = att_a.type(torch.float)
        att_b = att_b.type(torch.float)
        
        
        if (it+1) % (args.n_d+1) != 0:
            errD = attgan.trainD(img_a, att_a, att_b)
            # add_scalar_dict(writer, errD, it+1, 'D')
        else:
            errG = attgan.trainG(img_a, att_a, att_b)
            # add_scalar_dict(writer, errG, it+1, 'G')
            progressbar.say(epoch=epoch, iter=it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])
        
        if (it+1) % args.save_interval == 0:
            # To save storage space, I only checkpoint the weights of G.
            # If you'd like to keep weights of G, D, optim_G, optim_D,
            # please use save() instead of saveG().
            # attgan.saveG(os.path.join(
            #     'result', args.experiment, args.name, str(float(args.biased_var)), args.hyperparameter, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            # ))
            attgan.saveG(os.path.join(
                '/nas/vista-ssd01/users/jiazli/attGAN', args.experiment, args.name, str(float(args.biased_var)), args.hyperparameter, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            ))
            # attgan.save(os.path.join(
            #     'result', args.experiment_name, str(float(args.biased_var)), args.hyperparameter, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            # ))
        if (it+1) % args.sample_interval == 0:
            attgan.eval()
            with torch.no_grad():
                samples = [fixed_img_a]
                for i, att_b in enumerate(sample_att_b_list):
                    samples.append(attgan.G(fixed_img_a, att_b))
                samples = torch.cat(samples, dim=3)
                # writer.add_image('sample', vutils.make_grid(samples, nrow=1, normalize=True, range=(-1., 1.)), it+1)
                vutils.save_image(samples, os.path.join(
                        'result', args.experiment, args.name, str(float(args.biased_var)), args.hyperparameter, 'sample_training',
                        'Epoch_({:d})_({:d}of{:d}).jpg'.format(epoch, it%it_per_epoch+1, it_per_epoch)
                    ), nrow=1, normalize=False, range=(0., 1.))
                wandb.log({'test/filtered images': wandb.Image(vutils.make_grid(samples, nrow=1, padding=0, normalize=False))})
        it += 1

        