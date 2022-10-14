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
from models.CelebA_attgan_MI_Pwarmup import Model as C


class Model(C):

    def trainG(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = False
        
        zs_a = self.G(img_a, mode='enc')
        att_c_ = torch.zeros_like(att_b_)
        att_c = torch.ones_like(att_b_) / 2
        img_fake = self.G(zs_a, att_c_, mode='dec')
        img_recon = self.G(zs_a, att_a_, mode='dec')
        d_fake, dc_fake = self.D(img_fake), self.P(img_fake)
        
        if self.mode == 'wgan':    
            gf_loss = -d_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        gc_loss = F.binary_cross_entropy_with_logits(dc_fake, att_c)
        gr_loss = F.l1_loss(img_recon, img_a)
        a_tile = att_a_.view(att_a_.size(0), -1, 1, 1).repeat(1, self.dim_per_attr, self.f_size, self.f_size)
        z = zs_a[-1]
        mine_loss = -mi_criterion(z.view(z.size(0), -1), a_tile.detach().view(a_tile.size(0), -1), self.mine)
        # mine_loss = -mi_criterion(z.flatten(), a_tile.flatten(), self.mine)
        g_loss = gf_loss + self.gc * gc_loss + self.gr * gr_loss + self.mi * mine_loss
        
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()

        wandb.log({
            'g/total_loss': g_loss.item(),
            'g/fake_loss': gf_loss.item(),
            'g/classifier_loss': gc_loss.item(),
            'g/reconstuct_loss': gr_loss.item(),
            'g/mi_loss': mine_loss.item(),
            })
        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item()
        }
        return errG
    
    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = True
        
        att_c_ = torch.zeros_like(att_b_)
        att_c = torch.ones_like(att_b_) / 2
        img_fake = self.G(img_a, att_c_).detach()
        d_real, dc_real = self.D(img_a), self.P(img_a)
        d_fake, dc_fake = self.D(img_fake), self.P(img_fake)
        
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
        
        if self.mode == 'wgan': # discriminator becomes critic
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
                      F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        if self.mode == 'dcgan': # Deep Convolutional gan  # sigmoid_cross_entropy 
            df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                      F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        dc_loss = F.binary_cross_entropy_with_logits(dc_real, att_a)
        d_loss = df_loss + self.gp * df_gp + self.dc * dc_loss
        
        self.optim_D.zero_grad()
        self.optim_P.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        self.optim_P.step()
        
        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(), 
            'df_gp': df_gp.item(), 'dc_loss': dc_loss.item()
        }
        wandb.log({
            'd/total_loss': d_loss.item(),
            'd/fake_loss': df_loss.item(),
            'd/classifier_loss': dc_loss.item(),
            'd/df_gp_loss': df_gp.item(),
            })
        return errD