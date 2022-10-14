import os
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torchsummary import summary
from helpers import Progressbar
import wandb
import utils
import copy

from models.module import *
from models.module.MINE.model import M
from models.module.MINE.utils import mi_criterion
from dataloader.CelebA_ import check_attribute_conflict
from models.CMNIST_attgan_MI_Pwarmup_neutral import Model as A


class Model(A):
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.gr = args.gr
        self.gc = args.gc
        self.dc = args.dc
        self.gp = args.gp
        # self.ga = args.ga
        self.mi = args.mi
        self.num_iter_MI = args.num_iter_MI
        self.dim_per_attr = args.dim_per_attr
        self.dim_attrs = args.dim_per_attr * args.n_attrs
        self.f_size = args.img_size // 2**args.enc_layers  # f_size = 4 for 128x128
        
        self.hyperparameter = args.hyperparameter
        self.epoch = 0
        self.it = 0
        
        self.G = Generator_sigmoid(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size,
            args.dim_per_attr
        )
        self.G.train()
        if self.gpu: self.G.cuda()
        summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        self.D = Discriminator(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size,
            args.n_attrs
        )
        self.D.train()
        if self.gpu: self.D.cuda()
        summary(self.D, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        self.P = ColorPredictor()
        self.P.load_weights(file_path=os.path.join('/nas/home/jiazli/code/Adversarial-Filter-Debiasing/result/CMNIST_pretrain/1014/', str(float(args.biased_var)), '32_0.001_best.pth'))
        # self.P.train() # not working
        self.P.eval()
        if self.gpu: self.P.cuda()
        summary(self.P, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')

        # MI
        hidden_dim = [1024, 256, 64, 10]
        # print((args.enc_dim * 2 ** (args.enc_layers-1) + self.dim_attrs) * 4 * 4)
        self.mine = M(input_dim=(args.enc_dim * 2 ** (args.enc_layers-1) + self.dim_attrs) * 4 * 4, hidden_dim=hidden_dim) # 64 * (2**(3-1)) * 3 * 3 = 4096
        self.mine.train()
        if self.gpu: self.mine.cuda()
        summary(self.mine, [(1, (args.enc_dim * 2 ** (args.enc_layers-1) + self.dim_attrs) * 4 * 4)], batch_size=4, device='cuda' if args.gpu else 'cpu')

        if torch.cuda.device_count() > 1:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.P = nn.DataParallel(self.P)
            self.mine = nn.DataParallel(self.mine)
        
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
        # self.optim_P = optim.Adam(self.P.parameters(), lr=args.lr, betas=args.betas)
        self.optim_mine = optim.Adam(self.mine.parameters(), lr=args.lr, betas=args.betas)
    
    def train_epoch(self, train_dataloader, valid_dataloader, it_per_epoch, args):
        progressbar = Progressbar()
        # train with base lr in the first 100 epochs and half the lr in the last 100 epochs
        lr = args.lr_base / (10 ** (self.epoch // 20))
        self.set_lr(lr)
    
        # pretrain MINE at the beginning of every epoch
        mine_loader = copy.deepcopy(train_dataloader)
        batch_iter = iter(mine_loader)
        self.pretrainMI(mine_loader, args)

        # start iteration
        errG, errD = None, None
        for img_a, att_a in progressbar(train_dataloader):
            # prepare data
            img_a, att_a, att_a_, att_b, att_b_ = self.prepare_data(img_a, att_a, args)

            # train model
            self.train()

            if (self.it+1) % (args.n_d+1) != 0:
                errD = self.trainD(img_a, att_a, att_a_, att_b, att_b_)
            else:
                errG = self.trainG(img_a, att_a, att_a_, att_b, att_b_)
                batch_iter = self.trainMI(batch_iter, mine_loader, args)
            if errD and errG:
                progressbar.say(epoch=self.epoch, iter=self.it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])

            if (self.it+1) % args.save_interval == 0:
                self.save_model(args)
                self.eval_model(valid_dataloader, it_per_epoch, args)
            self.it += 1
        self.epoch += 1
        # renew mine for new epoch
        self.initMINE()
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
        # for g in self.optim_P.param_groups:
        #     g['lr'] = lr
        for g in self.optim_mine.param_groups:
            g['lr'] = lr
    
    def train(self):
        self.G.train()
        self.D.train()
        # self.P.train()
        self.mine.train()
    
    def eval(self):
        self.G.eval()
        self.D.eval()
        # self.P.eval()
        self.mine.eval()

    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            # 'P': self.P.state_dict(),
            'mine': self.mine.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict(),
            # 'optim_P': self.optim_P.state_dict(),
            'optim_mine': self.optim_mine.state_dict(),
        }
        torch.save(states, path)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        # if 'P' in states:
        #     self.P.load_state_dict(states['P'])
        if 'mine' in states:
            self.mine.load_state_dict(states['mine'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
        if 'optim_P' in states:
            self.optim_P.load_state_dict(states['optim_P'])
        if 'optim_mine' in states:
            self.optim_mine.load_state_dict(states['optim_mine'])

    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        # if 'P' in states:
        #     self.P.load_state_dict(states['P'])
        if 'mine' in states:
            self.mine.load_state_dict(states['mine'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
        # if 'optim_P' in states:
        #     self.optim_P.load_state_dict(states['optim_P'])
        if 'optim_mine' in states:
            self.optim_mine.load_state_dict(states['optim_mine'])