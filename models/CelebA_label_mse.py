"""EGAN"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import wandb

from backbones import *
from models.CelebA_label import Model as A

class Model(A):

    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.gr = args.gr
        self.gc = args.gc
        self.dc = args.dc
        self.gp = args.gp
        self.ga = args.ga
        self.dim_per_attr = args.dim_per_attr
        self.f_size = args.img_size // 2**args.enc_layers  # f_size = 4 for 128x128

        self.G = Generator_no_inject(
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

        self.P = Predictor()
        self.P.load_weights(file_path='/nas/home/jiazli/code/Adversarial-Filter-Debiasing/pretrain/predictor/CelebA/label.pth')
        self.P.eval()
        if self.gpu: self.P.cuda()
        summary(self.P, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.P = nn.DataParallel(self.P)
        
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
        self.optim_P = optim.Adam(self.P.parameters(), lr=args.lr, betas=args.betas)
    
    def _att_criterion(self, zs, a):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, self.dim_per_attr, self.f_size, self.f_size)
        # print(a_tile.size())
        # print(zs[-1][:,:self.dim_per_attr,:,:].size())
        # print('jasjodlas')
        return F.l1_loss(zs[-1][:,:self.dim_per_attr,:,:], a_tile)
    
    # def train():
    #     if (it+1) % (args.n_d+1) != 0:
    #         errD = attgan.trainD(img_a, att_a, att_b)
    #     else:
    #         errG = attgan.trainG_P1(img_a, att_a, att_b)
    #         errG = attgan.trainG_P2(img_a, att_a, att_b)
    #         progressbar.say(epoch=epoch, iter=it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])
    
    def trainG_P1(self, img_a, att_a, att_b):
        for p in self.D.parameters():
            p.requires_grad = False
        zs_a = self.G(img_a, mode='enc')
        att_loss = self._att_criterion(zs_a, att_a)
        
        img_recon = self.G(zs_a, att_a, mode='dec')
        d_recon, dc_recon = self.D(img_recon), self.P(img_recon)

        gc_loss = self.P._criterion_pred(dc_recon, att_a)
        gr_loss = F.l1_loss(img_recon, img_a)
        g_loss =  self.ga * att_loss + self.gc * gc_loss + self.gr * gr_loss

        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        wandb.log({
            'g1/total_loss': g_loss.item(),
            'g1/att_loss': att_loss.item(),
            'g1/classifier_loss': gc_loss.item(),
            'g1/reconstuct_loss': gr_loss.item(),
            })
        errG = {
            'g_loss': g_loss.item(), 
        }
        return errG 

    
    def trainG_P2(self, img_a, att_a, att_b):
        for p in self.D.parameters():
            p.requires_grad = False

        zs_a = self.G(img_a, mode='enc')
        img_fake = self.G(zs_a, att_b, mode='dec')
        img_recon = self.G(zs_a, att_a, mode='dec')
        d_fake, dc_fake = self.D(img_fake), self.P(img_fake)
        d_recon, dc_recon = self.D(img_recon), self.P(img_recon)

        if self.mode == 'wgan':
            gf_loss = -d_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(F.sigmoid(d_fake), torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        gc_loss = self.P._criterion_pred(dc_fake, att_b) + self.P._criterion_pred(dc_recon, att_a)
        gr_loss = F.l1_loss(img_recon, img_a)
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

    def trainD(self, img_a, att_a, att_b):
        for p in self.D.parameters():
            p.requires_grad = True
        
        # uniform label 0 is between 0.5 and -0.5
        att_c = torch.ones_like(att_b) / 2
        
        img_fake = self.G(img_a, att_b).detach()
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
        
        if self.mode == 'wgan':
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
                      F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                      F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        # print(dc_real.size())
        # print(att_a.size())
        dc_loss = self.P._criterion_pred(dc_real, att_a)
        d_loss = df_loss + self.gp * df_gp + self.dc * dc_loss
        
        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        
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
    
    def train(self):
        self.G.train()
        self.D.train()
    
    def eval(self):
        self.G.eval()
        self.D.eval()
    
    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
    
    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
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
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.n_attrs = 13
    args.betas = (args.beta1, args.beta2)
    attgan = AttGAN(args)
