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
import os
import torchvision.utils as vutils

from models.module import *
from dataloader.CelebA_ import check_attribute_conflict


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
        self.dim_attrs = args.dim_per_attr * args.n_attrs
        self.f_size = args.img_size // 2**args.enc_layers  # f_size = 4 for 128x128
        
        self.scheduler = itertools.cycle([0] * args.num_g1 + [1] * args.num_g2 + [2] * args.num_dis)
        self.hyperparameter = args.hyperparameter

        self.epoch = 0
        self.it = 0

        # Generator
        self.G = Generator_no_inject_sigmoid(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size,
            args.dim_per_attr)
        self.G.train()
        if self.gpu: self.G.cuda()
        summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        # Discriminator
        self.D = Discriminator(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size,
            args.n_attrs
        )
        self.D.train()
        if self.gpu: self.D.cuda()
        summary(self.D, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')

        self.P = Adv()
        self.P.load_weights(file_path=os.path.join('/nas/home/jiazli/code/Adversarial-Filter-Debiasing/pretrain/predictor/CMNIST/921', str(float(args.biased_var)), '32_0.001_best.pth'))
        self.P.eval()
        if self.gpu: self.P.cuda()
        summary(self.P, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')

        # MI
        self.mine = M(input_dim=args.enc_dim * 2 ** (args.enc_layers-1) * 4 * 4) # 64 * (2**(3-1)) * 3 * 3 = 4096
        self.mine.train()
        if self.gpu: self.mine.cuda()

        if torch.cuda.device_count() > 1:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.mine = nn.DataParallel(self.mine)
        
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
        self.optim_P = optim.Adam(self.P.parameters(), lr=args.lr, betas=args.betas)
        self.optim_mine = optim.Adam(self.mine.parameters(), lr=args.lr, betas=args.betas)

    # def renewMINE(self, args):
    #     some_module = self.mine.layer
    #     # del self.mine.layer
    #     del self.optim_mine
    #     del some_module

    #     self.mine = M(input_dim=args.enc_dim * 2 ** (args.enc_layers-1) * 7 * 7) # 64 * (2**(5-1)) * 7 * 7 = 50176
    #     self.mine.train()
    #     if self.gpu: self.mine.cuda()
    #     self.optim_mine = optim.Adam(self.mine.parameters(), lr=args.lr, betas=args.betas)
    
    def initMINE(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # m.weight.xavier_uniform_()
                m.bias.data.fill_(0.01)
        self.mine.apply(init_weights)
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
        for g in self.optim_mine.param_groups:
            g['lr'] = lr

    def _attr_criterion(self, zs, a, which_loss='l1'):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, self.dim_per_attr, self.f_size, self.f_size)
        if which_loss == 'mse':
            loss = F.mse_loss(zs[:,:self.dim_attrs,:,:], a_tile)
        elif which_loss == 'l1':
            loss = F.l1_loss(zs[:,:self.dim_attrs,:,:], a_tile)
        return loss
    
    def prepare_data(self, img_a, att_a, args):
        img_a = img_a.cuda() if args.gpu else img_a
        att_a = att_a.cuda() if args.gpu else att_a

        att_b = torch.ones_like(att_a)
        
        att_a = att_a.type(torch.float)
        att_b = att_b.type(torch.float)
        return img_a, att_a, att_a, att_b, att_b
    
    def _criterion_regr(self, output, target):
        # return F.l1_loss(output, target)
        return F.mse_loss(output, target)
    
    # def warmupP(self, P_loader, args):
    #     progressbar = Progressbar()
    #     loss_pred = 0
    #     pretrain_it = 0
    #     print('Pretrain ColorPredictor')
    #     for img_a, att_a in progressbar(P_loader):
    #         img_a, att_a, att_a_, att_b, att_b_ = self.prepare_data(img_a, att_a, args)
    #         self.optim_P.zero_grad()
    #         outputs = self.P(img_a)
    #         pred_loss = self._criterion_regr(outputs, att_a)
    #         pred_loss.backward()
    #         self.optim_P.step()
    #         loss_pred += pred_loss.item()
    #         wandb.log({'warmup/batch pred loss': pred_loss.item()})
    #         pretrain_it += 1
    #         progressbar.say(iter=pretrain_it+1, pred_loss = pred_loss.item())

    def pretrainMI(self, mine_loader, args):
        progressbar = Progressbar()
        loss_mine = 0
        pretrain_it = 0
        print('Pretrain MINE')
        for img_a, att_a in progressbar(mine_loader):
            img_a, att_a, att_a_, att_b, att_b_ = self.prepare_data(img_a, att_a, args)
            self.optim_mine.zero_grad()
            z = self.G(img_a, mode='enc').detach()
            z_a = z[:,:self.dim_attrs].view(z.size(0), -1)
            z_s = z[:,self.dim_attrs:].view(z.size(0), -1)
            # print(z_a.size())
            # print(z_s.size())
            mine_loss = mi_criterion(z_a, z_s, self.mine)
            mine_loss.backward()
            self.optim_mine.step()
            loss_mine += mine_loss.item()
            wandb.log({'pretrain/batch mine loss': mine_loss.item()})
            pretrain_it += 1
            progressbar.say(iter=pretrain_it+1, mine_loss = mine_loss.item())
    
    def train_epoch(self, train_dataloader, valid_dataloader, it_per_epoch, args):
        progressbar = Progressbar()
        # train with base lr in the first 20 epochs and half the lr in the last 100 epochs
        lr = args.lr_base / (10 ** (self.epoch // 20))
        self.set_lr(lr)

        # pretrain MINE at the beginning of every epoch
        mine_loader = copy.deepcopy(train_dataloader)
        batch_iter = iter(mine_loader)
        self.pretrainMI(mine_loader, args)

        # # pretrain Predictor at the beginning of every epoch
        # pred_loader = copy.deepcopy(train_dataloader)
        # for _ in range(10):
        #     self.warmupP(pred_loader, args)
        
        # start iteration
        errG1, errG2, errD = None, None, None
        for img_a, att_a in progressbar(train_dataloader):
            # prepare data
            img_a, att_a, att_a_, att_b, att_b_ = self.prepare_data(img_a, att_a, args)

            # train model
            phase = next(self.scheduler)
            self.train()
            if phase == 0:
                # train encoder and decoder, let specific dimension of latent vector to represent sex
                errG1 = self.trainG_P1(img_a, att_a, att_a_, att_b, att_b_)
                # retrain MINE to be familiar with new representation
                batch_iter = self.trainMI(batch_iter, mine_loader, args)
            elif phase == 1:
                # train decoder to express the wanted sex attribute from specific dimension
                errG2 = self.trainG_P2(img_a, att_a, att_a_, att_b, att_b_)
            elif phase == 2:
                # 1. train classifier to well predict sex; 2. train discriminator to well classify real and fake image
                errD = self.trainD(img_a, att_a, att_a_, att_b, att_b_)
            if errD and errG1 and errG2:
                progressbar.say(epoch=self.epoch, iter=self.it+1, d_loss=errD['d_loss'], g1_loss=errG1['g_loss'] , g2_loss=errG2['g_loss'])

            # if phase == 0 or phase == 1:
            #     # train encoder and decoder, let specific dimension of latent vector to represent sex
            #     errG1 = self.trainG_P1(img_a, att_a, att_a_, att_b, att_b_)
            #     # retrain MINE to be familiar with new representation
            #     batch_iter = self.trainMI(batch_iter, mine_loader, args)
            #     # train decoder to express the wanted sex attribute from specific dimension
            #     errG2 = self.trainG_P2(img_a, att_a, att_a_, att_b, att_b_)
            # elif phase == 2:
            #     # 1. train classifier to well predict sex; 2. train discriminator to well classify real and fake image
            #     errD = self.trainD(img_a, att_a, att_a_, att_b, att_b_)
            # if errD and errG1 and errG2:
            #     progressbar.say(epoch=self.epoch, iter=self.it+1, d_loss=errD['d_loss'], g1_loss=errG1['g_loss'] , g2_loss=errG2['g_loss'])

            if (self.it+1) % args.save_interval == 0:
                self.save_model(args)
                self.eval_model(valid_dataloader, it_per_epoch, args)
            self.it += 1
        self.epoch += 1
        # renew mine for new epoch
        self.initMINE()
    
    def trainMI(self, batch_iter, mine_loader, args):
        for i in range(self.num_iter_MI):
            img_a, att_a, batch_iter = utils.nextbatch(batch_iter, mine_loader)
            img_a, att_a, att_a_, att_b, att_b_ = self.prepare_data(img_a, att_a, args)
            self.optim_mine.zero_grad()
            z = self.G(img_a, mode='enc').detach()
            z_a = z[:,:self.dim_attrs].view(z.size(0), -1)
            z_s = z[:,self.dim_attrs:].view(z.size(0), -1)
            mine_loss = mi_criterion(z_a, z_s, self.mine)
            mine_loss.backward()
            self.optim_mine.step()
            wandb.log({'midtrain/batch mine loss': mine_loss.item()})
        return batch_iter
        
    def trainG_P1(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = False

        z = self.G(img_a, mode='enc')
        img_recon = self.G(z, att_a_, mode='dec')
        d_recon, dc_recon = self.D(img_recon), self.P(img_recon)

        # if self.mode == 'wgan':
        #     gf_loss = -d_recon.mean()
        # if self.mode == 'lsgan':  # mean_squared_error
        #     gf_loss = F.mse_loss(F.sigmoid(d_fake), torch.ones_like(d_fake))
        # if self.mode == 'dcgan':  # sigmoid_cross_entropy
        #     gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

        # 1. maintain the reconstruct quality of decoder output
        # print(img_recon)
        # print(img_a)
        gr_loss = F.l1_loss(img_recon, img_a)
        # gr_loss = F.mse_loss(img_recon, img_a)

        # 2. classify the sex out from reconstruct image, let recon image yield such sex
        # gc_loss = F.binary_cross_entropy_with_logits(dc_recon, att_a)
        gc_loss = F.mse_loss(dc_recon, att_a)

        # 3. let specific dimension to be sex itself
        ga_loss = self._attr_criterion(z, att_a_)

        # 4. let the other dimension to be independent with sex dimension
        z_a = z[:,:self.dim_attrs].detach().view(z.size(0), -1)
        z_s = z[:,self.dim_attrs:].detach().view(z.size(0), -1)
        mine_loss = -mi_criterion(z_a, z_s, self.mine) # minimize mi to update model

        g_loss = self.ga * ga_loss + self.gc * gc_loss + self.gr * gr_loss + self.mi * mine_loss

        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        wandb.log({
            'g1/total_loss': g_loss.item(),
            # 'g1/fake_loss': gf_loss.item(),
            'g1/reconstuct_loss': gr_loss.item(),
            'g1/classifier_loss': gc_loss.item(),
            'g1/attribute_loss': ga_loss.item(),
            'g1/mine_loss': mine_loss.item(),
            })
        errG = {
            'g_loss': g_loss.item(), 
        }
        return errG 
    
    def trainG_P2(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = False

        z = self.G(img_a, mode='enc').detach()
        img_fake = self.G(z, att_b_, mode='dec_erase')
        img_recon = self.G(z, att_a_, mode='dec_erase')
        d_fake, dc_fake = self.D(img_fake), self.P(img_fake)
        d_recon, dc_recon = self.D(img_recon), self.P(img_recon)

        # 1. let image with wanted attribute to be real
        if self.mode == 'wgan':
            gf_loss = -d_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(F.sigmoid(d_fake), torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        
        # 2. classify the sex out from reconstruct image, let fake image yield such sex
        # gc_fake_loss = F.binary_cross_entropy_with_logits(dc_fake, att_b) 
        gc_fake_loss = F.mse_loss(dc_fake, att_b) 

        # 3. maintain the reconstruct quality of decoder output with perturb dimension
        gr_loss = F.l1_loss(img_recon, img_a)
        # gr_loss = F.mse_loss(img_recon, img_a)

        # 4. classify the sex out from reconstruct image, let recon image yield such sex
        # gc_recon_loss = F.binary_cross_entropy_with_logits(dc_recon, att_a)
        gc_recon_loss = F.mse_loss(dc_recon, att_a)

        g_loss = gf_loss + self.gc * gc_fake_loss + self.gc * gc_recon_loss + self.gr * gr_loss
        
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        wandb.log({
            'g2/total_loss': g_loss.item(),
            'g2/fake_loss': gf_loss.item(),
            'g2/classifier_fake_loss': gc_fake_loss.item(),
            'g2/classifier_recon_loss': gc_recon_loss.item(),
            'g2/reconstuct_loss': gr_loss.item(),
            })
        errG = {
            'g_loss': g_loss.item(), 
            # 'gf_loss': gf_loss.item(),
            # 'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item()
        }
        return errG

    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = True

        d_real, dc_real = self.D(img_a), self.P(img_a)
        img_fake = self.G(img_a, att_b_).detach() # erase
        img_recon = self.G(img_a, att_a_).detach() # erase
        d_fake, dc_fake = self.D(img_fake), self.P(img_fake)
        d_recon, dc_recon = self.D(img_recon), self.P(img_recon)
        
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
        
        # 1. let image with wanted attribute to be real
        if self.mode == 'wgan':
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
            wd_recon = d_real.mean() - d_recon.mean()
            df_loss_recon = -wd_recon
            df_gp_recon = gradient_penalty(self.D, img_a, img_recon)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
                      F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                      F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)

        # 2. classify the sex out from fake image, let fake image yield such sex
        # dc_fake_loss = F.binary_cross_entropy_with_logits(dc_fake, att_b) 
        dc_fake_loss = F.mse_loss(dc_fake, att_b) 

        # 3. classify the sex out from reconstruct image, let recon image yield such sex
        # dc_recon_loss = F.binary_cross_entropy_with_logits(dc_recon, att_a)
        dc_recon_loss = F.mse_loss(dc_recon, att_a)

        # 3. classify the sex out from real image, train classifier
        # dc_real_loss = F.binary_cross_entropy_with_logits(dc_real, att_a)
        dc_real_loss = F.mse_loss(dc_real, att_a)

        d_loss = df_loss + self.gp * df_gp + self.dc * dc_fake_loss + self.dc * dc_recon_loss + self.dc * dc_real_loss
        # d_loss = df_loss + df_loss_recon + self.gp * (df_gp + df_gp_recon) + self.dc * dc_fake_loss + self.dc * dc_recon_loss
        
        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        
        wandb.log({
            'd/total_loss': d_loss.item(),
            'd/fake_loss': df_loss.item(),
            # 'd/df_gp_loss': df_gp.item(),
            # 'd/fake_loss_recon': df_loss_recon.item(),
            # 'd/df_gp_loss_recon': df_gp_recon.item(),
            'd/classifier_fake_loss': dc_fake_loss.item(),
            'd/classifier_recon_loss': dc_recon_loss.item(),
            'd/classifier_real_loss': dc_real_loss.item(),
            })

        errD = {
            'd_loss': d_loss.item(), 
            'df_loss': df_loss.item(), 
            'df_gp': df_gp.item(), 
            # 'dc_loss': dc_loss.item()
        }
        return errD
    
    def save_model(self, args):
        # save model
        # To save storage space, I only checkpoint the weights of G.
        # If you'd like to keep weights of G, D, optim_G, optim_D,
        # please use save() instead of saveG().
        self.saveG(os.path.join(
            '/nas/vista-ssd01/users/jiazli/attGAN', args.experiment, args.name, str(float(args.biased_var)), self.hyperparameter, 'checkpoint', 'weights.{:d}.pth'.format(self.epoch)
        ))
        # self.save(os.path.join(
        #     'result', args.experiment, args.name, hyperparameter, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
        # ))
    
    def eval_model(self, valid_dataloader, it_per_epoch, args):
        fixed_img_a, fixed_att_a = next(iter(valid_dataloader))
        fixed_img_a = fixed_img_a.cuda() if args.gpu else fixed_img_a
        fixed_att_a = fixed_att_a.cuda() if args.gpu else fixed_att_a
        sample_att_b_list = [fixed_att_a, torch.ones_like(fixed_att_a), torch.zeros_like(fixed_att_a)]

        # eval model
        self.eval()
        with torch.no_grad():
            samples = [fixed_img_a]
            for i, att_b in enumerate(sample_att_b_list):
                samples.append(self.G(fixed_img_a, att_b))
            samples = torch.cat(samples, dim=3)
            vutils.save_image(samples, os.path.join(
                    'result', args.experiment, args.name, str(float(args.biased_var)), self.hyperparameter, 'sample_training',
                    'Epoch_({:d})_({:d}of{:d}).jpg'.format(self.epoch, self.it%it_per_epoch+1, it_per_epoch)
                ), nrow=1, normalize=False, range=(0., 1.))
            # wandb.log({'test/filtered images': wandb.Image(vutils.make_grid(samples, nrow=1, padding=0, normalize=False))})
    
    def train(self):
        self.G.train()
        self.D.train()
        self.P.train()
        self.mine.train()
    
    def eval(self):
        self.G.eval()
        self.D.eval()
        self.P.eval()
        self.mine.eval()

    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'mine': self.mine.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict(),
            'optim_mine': self.optim_mine.state_dict()
        }
        torch.save(states, path)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'mine' in states:
            self.mine.load_state_dict(states['mine'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
        if 'optim_mine' in states:
            self.optim_mine.load_state_dict(states['optim_mine'])