import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import wandb
from helpers import Progressbar
import itertools
from models.module.MINE.model import M
from models.module.MINE.utils import mi_criterion

import utils
import copy

from models.module import *
from models.CelebA_label_mse_MI import Model as C

class Model(C):

    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.gr = args.gr
        self.gc = args.gc
        self.dc = args.dc
        self.gp = args.gp
        self.ga = args.ga
        self.mi = args.mi
        self.num_iter_MI = args.num_iter_MI
        self.dim_per_attr = args.dim_per_attr
        self.f_size = args.img_size // 2**args.enc_layers  # f_size = 4 for 128x128
        
        self.scheduler = itertools.cycle([0] * args.num_ganp1 + [1] * args.num_ganp2 + [2] * args.num_dis)
        self.hyperparameter = args.hyperparameter

        self.epoch = 0
        self.it = 0

        # Generator
        self.G = Generator_no_inject(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size,
            args.dim_per_attr)
        self.G.train()
        if self.gpu: self.G.cuda()
        summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        # Discriminator
        self.D = Discriminator_Predictor(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size,
            args.n_attrs
        )
        self.D.train()
        if self.gpu: self.D.cuda()
        summary(self.D, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')

        # MI
        self.mine = M(input_dim=args.enc_dim * 2 ** (args.enc_layers-1) * 7 * 7)
        self.mine.train()
        if self.gpu: self.mine.cuda()
        
        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            # self.P = nn.DataParallel(self.P)
            self.mine = nn.DataParallel(self.mine)
        
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
        # self.optim_P = optim.Adam(self.P.parameters(), lr=args.lr, betas=args.betas)
        self.optim_mine = optim.Adam(self.mine.parameters(), lr=args.lr, betas=args.betas)
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
        # for g in self.optim_P.param_groups:
        #     g['lr'] = lr
        for g in self.optim_mine.param_groups:
            g['lr'] = lr
    
    def train_epoch(self, train_dataloader, valid_dataloader, it_per_epoch, args):
        progressbar = Progressbar()
        # train with base lr in the first 100 epochs and half the lr in the last 100 epochs
        lr = args.lr_base / (10 ** (self.epoch // 100))
        self.set_lr(lr)

        mine_loader = copy.deepcopy(train_dataloader)
        batch_iter = iter(mine_loader)

        # pretrain MINE at the beginning of every epoch
        self.pretrainMI(mine_loader, args)
        
        # start iteration
        errG1, errG2, errD = None, None, None
        for img_a, att_a in progressbar(train_dataloader):
            # prepare data
            img_a, att_a, att_a_, att_b, att_b_ = self.prepare_data(img_a, att_a, args)

            # if self.G1pretrain != 0:
            #     errG1 = self.trainG_P1(img_a, att_a, att_a_, att_b, att_b_)
            #     self.G1pretrain -= 1

            # train model
            phase = next(self.scheduler)
            self.train()
            if phase == 0:
                errG1 = self.trainG_P1(img_a, att_a, att_a_, att_b, att_b_)
                batch_iter = self.trainMI(batch_iter, mine_loader, args)
            elif phase == 1:
                errG2 = self.trainG_P2(img_a, att_a, att_a,  att_b, att_b_)
            elif phase == 2:
                errD = self.trainD(img_a, att_a, att_a_, att_b, att_b_)
            if errD and errG1 and errG2:
                progressbar.say(epoch=self.epoch, iter=self.it+1, d_loss=errD['d_loss'], g1_loss=errG1['g_loss'] , g2_loss=errG2['g_loss'])

            self.save_model(args)
            self.eval_model(valid_dataloader, it_per_epoch, args)
            self.it += 1
        self.epoch += 1

    def trainMI(self, batch_iter, mine_loader, args):
        for i in range(self.num_iter_MI):
            img_a, att_a, batch_iter = utils.nextbatch(batch_iter, mine_loader)
            img_a, att_a, att_a_, att_b, att_b_ = self.prepare_data(img_a, att_a, args)
            self.optim_mine.zero_grad()
            zs_a = self.G(img_a, mode='enc')
            mine_loss = mi_criterion(zs_a[-1][:,:self.dim_per_attr].detach().view(zs_a[-1].size(0), -1), zs_a[-1][:,self.dim_per_attr:].detach().view(zs_a[-1].size(0), -1), self.mine)
            mine_loss.backward()
            self.optim_mine.step()
        return batch_iter
        
    def trainG_P1(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = False

        zs_a = self.G(img_a, mode='enc')
        img_recon = self.G(zs_a[-1], att_a_, mode='dec')
        d_recon, dc_recon = self.D(img_recon)

        if self.mode == 'wgan':
            gf_loss = -d_recon.mean()
        # if self.mode == 'lsgan':  # mean_squared_error
        #     gf_loss = F.mse_loss(F.sigmoid(d_fake), torch.ones_like(d_fake))
        # if self.mode == 'dcgan':  # sigmoid_cross_entropy
        #     gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        ga_loss = self._attr_criterion(zs_a[-1], att_a_)
        gc_loss = F.binary_cross_entropy_with_logits(dc_recon, att_a)
        gr_loss = F.l1_loss(img_recon, img_a)
        # gr_loss = F.mse_loss(img_recon, img_a)
        mine_loss = mi_criterion(zs_a[-1][:,:self.dim_per_attr].detach().view(zs_a[-1].size(0), -1), zs_a[-1][:,self.dim_per_attr:].detach().view(zs_a[-1].size(0), -1), self.mine)

        g_loss =  gf_loss + self.ga * ga_loss + self.gc * gc_loss + self.gr * gr_loss + self.mi * mine_loss

        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        wandb.log({
            'g1/total_loss': g_loss.item(),
            'g1/fake_loss': gf_loss.item(),
            'g1/attribute_loss': ga_loss.item(),
            'g1/classifier_loss': gc_loss.item(),
            'g1/reconstuct_loss': gr_loss.item(),
            'g1/mine_loss': mine_loss.item(),
            })
        errG = {
            'g_loss': g_loss.item(), 
        }
        return errG 
    
    def trainG_P2(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = False

        zs_a = self.G(img_a, mode='enc')
        img_fake = self.G(zs_a[-1].detach(), att_b_, mode='dec_erase')
        img_recon = self.G(zs_a[-1].detach(), att_a_, mode='dec_erase')
        d_fake, dc_fake = self.D(img_fake)
        d_recon, dc_recon = self.D(img_recon)

        if self.mode == 'wgan':
            gf_loss = -d_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(F.sigmoid(d_fake), torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        gc_loss = F.binary_cross_entropy_with_logits(dc_fake, att_b) + F.binary_cross_entropy_with_logits(dc_recon, att_a)
        gr_loss = F.l1_loss(img_recon, img_a)
        # gr_loss = F.mse_loss(img_recon, img_a)
        g_loss = gf_loss + self.gc * gc_loss + self.gr * gr_loss
        
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        wandb.log({
            'g2/total_loss': g_loss.item(),
            'g2/fake_loss': gf_loss.item(),
            'g2/classifier_loss': gc_loss.item(),
            'g2/reconstuct_loss': gr_loss.item(),
            })
        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item()
        }
        return errG

    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = True
        
        # # uniform label
        # att_c = torch.ones_like(att_b) / 2
        
        d_real, dc_real = self.D(img_a)
        img_fake = self.G(img_a, att_b_).detach()
        d_fake, dc_fake = self.D(img_fake)
        img_recon = self.G(img_a, att_a_).detach()
        d_recon, dc_recon = self.D(img_recon)
        
        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp
        
        if self.mode == 'wgan':
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
            wd_recon = d_real.mean() - d_recon.mean()
            df_loss_recon = -wd_recon
            df_gp_recon = gradient_penalty(self.D, img_a, img_recon)
        # if self.mode == 'lsgan':  # mean_squared_error
        #     df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
        #               F.mse_loss(d_fake, torch.zeros_like(d_fake))
        #     df_gp = gradient_penalty(self.D, img_a)
        # if self.mode == 'dcgan':  # sigmoid_cross_entropy
        #     df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
        #               F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
        #     df_gp = gradient_penalty(self.D, img_a)
        # d_loss = df_loss + self.gp * df_gp
        dc_loss = F.binary_cross_entropy_with_logits(dc_fake, att_b) + F.binary_cross_entropy_with_logits(dc_recon, att_a)
        d_loss = df_loss + df_loss_recon + self.gp * (df_gp + df_gp_recon) + self.dc * dc_loss
        
        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        
        errD = {
            'd_loss': d_loss.item(), 
            'df_loss': df_loss.item(), 
            'df_gp': df_gp.item(), 
            # 'dc_loss': dc_loss.item()
        }
        wandb.log({
            'd/total_loss': d_loss.item(),
            'd/fake_loss': df_loss.item(),
            'd/df_gp_loss': df_gp.item(),
            'd/fake_loss_recon': df_loss_recon.item(),
            'd/df_gp_loss_recon': df_gp_recon.item(),
            })
        return errD
    
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
            'optim_mine': self.optim_mine.state_dict()
        }
        torch.save(states, path)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        # if 'P' in states:
            # self.P.load_state_dict(states['D'])
        if 'mine' in states:
            self.mine.load_state_dict(states['mine'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
        # if 'optim_P' in states:
            # self.optim_P.load_state_dict(states['optim_D'])
        if 'optim_mine' in states:
            self.optim_mine.load_state_dict(states['optim_mine'])