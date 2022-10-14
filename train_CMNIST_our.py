"""Main entry point for training AttGAN network."""

import argparse
import datetime
import json
import os
from os.path import join
import torch
import torch.utils.data as data
import wandb
from dataloader.CMNIST import ColoredDataset_generated
from torchvision import datasets, transforms

torch.cuda.empty_cache()

def parse(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', dest='data', type=str, default='CMNIST')
    parser.add_argument('--data_path', dest='data_path', type=str, default="/nas/vista-ssd01/users/jiazli/datasets/MNIST")
    parser.add_argument("--biased_var", type=float, default=-1) # -1 uniformly
    
    parser.add_argument('--img_size', dest='img_size', type=int, default=32)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=0) #
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0) #
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
    parser.add_argument('--gr', dest='gr', type=float, default=100.0) # TODO matter pixel similarity
    parser.add_argument('--gc', dest='gc', type=float, default=300.0) # TODO
    parser.add_argument('--dc', dest='dc', type=float, default=1.0) # TODO
    parser.add_argument('--gp', dest='gp', type=float, default=10.0) # default in WGAN-GP
    parser.add_argument('--ga', dest='ga', type=float, default=5.0) # TODO
    parser.add_argument('--mi', dest='mi', type=float, default=5.0) # TODO
    parser.add_argument('--num_iter_MI', type=int, default=20) # TODO
    # parser.add_argument('--gf', dest='gf', type=float, default=1.0) #
    parser.add_argument('--dim_per_attr', type=int, default=10) # TODO matters serious
    parser.add_argument('--num_g1', type=int, default=1) # TODO
    parser.add_argument('--num_g2', type=int, default=1) # TODO 
    parser.add_argument('--num_dis', type=int, default=3) # TODO matter style
    
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan']) #
    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=4)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5) # default in WGAN-GP
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999) # default in WGAN-GP
    # parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
    
    # parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')
    
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--no_gpu', dest='gpu', action='store_false')
    # parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument("--experiment", metavar="",)
    parser.add_argument('--name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    
    return parser.parse_args(args)

args = parse()


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    args.batch_size *= (torch.cuda.device_count() - 1)
    args.lr *= (torch.cuda.device_count() - 1)
print(args)

args.lr_base = args.lr
args.n_attrs = 3 # RGB
args.betas = (args.beta1, args.beta2)
# args.hyperparameter = f'shortcut{args.shortcut_layers}-inject{args.inject_layers}-gr{args.gr}-gc{args.gc}-dc{args.dc}-gp{args.gp}-ga{args.ga}-mi{args.mi}'
args.hyperparameter = f'gr{args.gr}-gc{args.gc}-dc{args.dc}-ga{args.ga}-mi{args.mi}-dpa{args.dim_per_attr}-{args.num_g1}-{args.num_g2}-{args.num_dis}'


wandb.init(project="AttGAN",
            entity="jiazhi", 
            config=args, 
            group=args.experiment, 
            job_type=args.name, 
            name=args.hyperparameter
)

os.makedirs(join('result', args.experiment, args.name, str(args.biased_var), args.hyperparameter), exist_ok=True)
# os.makedirs(join('result', args.experiment, args.name, str(args.biased_var), args.hyperparameter, 'checkpoint'), exist_ok=True)
os.makedirs(join('/nas/vista-ssd01/users/jiazli/attGAN', args.experiment, args.name, str(args.biased_var), args.hyperparameter, 'checkpoint'), exist_ok=True)
os.makedirs(join('result', args.experiment, args.name, str(args.biased_var), args.hyperparameter, 'sample_training'), exist_ok=True)
with open(join('result', args.experiment, args.name, str(args.biased_var), args.hyperparameter, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))
with open(join('/nas/vista-ssd01/users/jiazli/attGAN', args.experiment, args.name, str(args.biased_var), args.hyperparameter, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

# choose model
if args.experiment == 'label':
    from models.CelebA_label import Model 
    m = Model(args)
elif args.experiment == 'label_mse':
    from models.CelebA_label_mse import Model
    m = Model(args)
# elif args.experiment == 'label_mse_G1D':
#     from models.CelebA_label_mse_G1D import Model
#     m = Model(args)
# elif args.experiment == 'label_mse_G1pretrain':
#     from models.CelebA_label_mse_G1pretrain import Model
#     args.G1pretrain = 100
#     m = Model(args)
# elif args.experiment == 'label_mse_MI':
#     from models.CelebA_label_mse_MI import Model
#     m = Model(args)
# elif args.experiment == 'label_mse_MI_Pscratch':
#     from models.CelebA_label_mse_MI_Pscratch import Model
#     m = Model(args)
# elif args.experiment == 'attGAN_MI':
#     from models.CelebA_attgan_MI import Model
#     m = Model(args)
# elif args.experiment == 'attGAN':
#     from models.CelebA_attgan import Model
#     m = Model(args)
elif args.experiment == 'label_mse_MI_PDshare':
    from models.CMNIST_label_mse_MI_PDshare import Model
    m = Model(args)
elif args.experiment == 'label_mse_MI_Pwarmup':
    from models.CMNIST_label_mse_MI_Pwarmup import Model
    m = Model(args)
elif args.experiment == 'label_mse_MI_pretrain':
    from models.CMNIST_label_mse_MI_pretrain import Model
    m = Model(args)



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










# start epoch
it_per_epoch = len(train_dataset) // args.batch_size
for epoch in range(args.epochs):
    m.train_epoch(train_dataloader, valid_dataloader, it_per_epoch, args)