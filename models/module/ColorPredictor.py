from asyncore import file_dispatcher
import torch
import torch.nn as nn
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from models.module.Encoder import *
from models.module.Regressor import *
from torchsummary import summary
import wandb
import os


class Adv(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.hidden_size = 128
        # ========= create models ===========
        self.encoder = Encoder(e_dim=self.hidden_size)
        self.regressor = Regressor(e_dim=self.hidden_size)
        self.sig = nn.Sigmoid()
    
    def load_weights(self, file_path, optim_pred=None):
        ckpt = torch.load(file_path)
        self.epoch = ckpt['pretrain_epoch']
        self.encoder.load_state_dict(ckpt["encoder"])
        self.regressor.load_state_dict(ckpt["regressor"])
        if optim_pred is not None:
            optim_pred_state_dict = ckpt["optim_pred"]
            if optim_pred_state_dict is None:
                print("WARNING: No optim_pred state dict found")
            else:
                optim_pred.load_state_dict(optim_pred_state_dict)
    
    # def _criterion_regr(self, output, target):
    #     # return F.l1_loss(output, target)
    #     return F.mse_loss(output, target)
    
    def forward(self, x):
        z = self.encoder(x)
        a_pred = self.regressor(z)
        return self.sig(a_pred)