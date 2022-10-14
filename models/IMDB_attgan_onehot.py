import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.utils as vutils
from torchsummary import summary
from helpers import Progressbar
import itertools
import wandb
import utils
import os

from models.module import *



class Model():
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
        
        self.G = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size,
            args.dim_per_attr
        )
        self.G.train()
        if self.gpu: self.G.cuda()
        summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        self.D = Discriminator_Predictor_IMDB(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size,
            args.n_attrs
        )
        self.D.train()
        if self.gpu: self.D.cuda()
        summary(self.D, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
    
    def prepare_data(self, img_a, att_a, args):
        att_a = torch.unsqueeze(att_a,1) if len(list(att_a.size())) == 1 else att_a
        img_a = img_a.cuda() if args.gpu else img_a
        att_a = att_a.cuda() if args.gpu else att_a

        
        

        att_b = torch.ones_like(att_a) * 6

        

        att_a = att_a.type(torch.float)
        att_b = att_b.type(torch.float)

        
        # onehot encoding
        att_a_ = utils.one_hot_embedding(att_a, 12)
        att_b_ = utils.one_hot_embedding(att_b, 12)
        att_a_ = att_a_.cuda() if args.gpu else att_a_
        att_b_ = att_b_.cuda() if args.gpu else att_b_

        # att_a_ = (att_a * 2 - 1) * args.thres_int # -1/2, 1/2 for all
        # att_b_ = (att_b * 2 - 1) * args.thres_int # -1/2, 1/2 for all
        return img_a, att_a, att_a_, att_b, att_b_
    
    def train_epoch(self, train_dataloader, valid_dataloader, it_per_epoch, args):
        progressbar = Progressbar()
        # train with base lr in the first 100 epochs and half the lr in the last 100 epochs
        lr = args.lr_base / (10 ** (self.epoch // 100))
        self.set_lr(lr)
        
        # start iteration
        errG, errD = None, None
        for img_a, sex_a, age_a in progressbar(train_dataloader):
            # prepare data
            # print(age_a)
            # print(age_a.size())
            img_a, att_a, att_a_, att_b, att_b_ = self.prepare_data(img_a, age_a, args)

            # # train model
            self.train()

            if (self.it+1) % (args.n_d+1) != 0:
                errD = self.trainD(img_a, att_a, att_a_, att_b, att_b_)
            else:
                errG = self.trainG(img_a, att_a, att_a_, att_b, att_b_)
            if errD and errG:
                progressbar.say(epoch=self.epoch, iter=self.it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])

            self.save_model(args)
            self.eval_model(valid_dataloader, it_per_epoch, args)
            self.it += 1
        self.epoch += 1
    
    
    def trainG(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = False
        
        zs_a = self.G(img_a, mode='enc')
        img_fake = self.G(zs_a, att_b_, mode='dec')
        img_recon = self.G(zs_a, att_a_, mode='dec')
        # img_fake = self.G(zs_a, att_b, mode='dec')
        # img_recon = self.G(zs_a, att_a, mode='dec')
        d_fake, dc_fake = self.D(img_fake)
        
        if self.mode == 'wgan':    
            gf_loss = -d_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        L = nn.CrossEntropyLoss()
        gc_loss = L(dc_fake, att_b.type(torch.long).squeeze())
        gr_loss = F.l1_loss(img_recon, img_a)
        g_loss = gf_loss + self.gc * gc_loss + self.gr * gr_loss
        
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()

        wandb.log({
            'g/total_loss': g_loss.item(),
            'g/fake_loss': gf_loss.item(),
            'g/classifier_loss': gc_loss.item(),
            'g/reconstuct_loss': gr_loss.item(),
            })
        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item()
        }
        return errG
    
    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = True
        
        img_fake = self.G(img_a, att_b_).detach()
        # img_fake = self.G(img_a, att_b).detach()
        d_real, dc_real = self.D(img_a)
        d_fake, dc_fake = self.D(img_fake)
        
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
        L = nn.CrossEntropyLoss()
        # print(dc_real.size())
        # print(att_a.size())
        # print('dsjalasj')
        dc_loss = L(dc_real, att_a.type(torch.long).squeeze())
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
            'optim_D': self.optim_D.state_dict(),
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
    
    
    def save_model(self, args):
        # save model
        if (self.it+1) % args.save_interval == 0:
            # To save storage space, I only checkpoint the weights of G.
            # If you'd like to keep weights of G, D, optim_G, optim_D,
            # please use save() instead of saveG().
            self.saveG(os.path.join(
                '/nas/vista-ssd01/users/jiazli/attGAN', args.experiment, args.name, self.hyperparameter, 'checkpoint', 'weights.{:d}.pth'.format(self.epoch)
            ))
            # self.save(os.path.join(
            #     'result', args.experiment, args.name, hyperparameter, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            # ))
    
    def eval_model(self, valid_dataloader, it_per_epoch, args):
        fixed_img_a, _, fixed_att_a = next(iter(valid_dataloader))


        fixed_img_a = fixed_img_a.cuda() if args.gpu else fixed_img_a
        fixed_att_a = fixed_att_a.cuda() if args.gpu else fixed_att_a
        fixed_att_a = fixed_att_a.type(torch.float)

        sample_att_b_list = [fixed_att_a, torch.ones_like(fixed_att_a) * 0, torch.ones_like(fixed_att_a) * 6, torch.ones_like(fixed_att_a) * 11]


        # eval model
        if (self.it+1) % args.sample_interval == 0:
            self.eval()
            with torch.no_grad():
                samples = [fixed_img_a]
                for i, att_b in enumerate(sample_att_b_list):
                    att_b_ = utils.one_hot_embedding(att_b, 12)
                    att_b_ = att_b_.cuda() if args.gpu else att_b_
                    samples.append(self.G(fixed_img_a, att_b))
                samples = torch.cat(samples, dim=3)
                vutils.save_image(samples, os.path.join(
                        'result', args.experiment, args.name, self.hyperparameter, 'sample_training',
                        'Epoch_({:d})_({:d}of{:d}).jpg'.format(self.epoch, self.it%it_per_epoch+1, it_per_epoch)
                    ), nrow=1, normalize=False, range=(0., 1.))
                # wandb.log({'test/filtered images': wandb.Image(vutils.make_grid(samples, nrow=1, padding=0, normalize=False))})
        
    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)