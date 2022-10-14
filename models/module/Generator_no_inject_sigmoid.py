from asyncore import file_dispatcher
import torch
import torch.nn as nn
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from torchsummary import summary
import wandb

MAX_DIM = 64 * 16  # 1024

class Generator_no_inject_sigmoid(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=1, shortcut_layers=1, inject_layers=1, img_size=224, dim_per_attr = 5):
        super(Generator_no_inject_sigmoid, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128
        self.dim_per_attr = dim_per_attr
        
        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)
        # for i in range(enc_layers):
        #     if i < enc_layers - 1:
        #         n_out = min(enc_dim * 2**i, MAX_DIM)
        #         layers += [Conv2dBlock(
        #             n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
        #         )]
        #         n_in = n_out
        #     else:
        #         n_out = min(enc_dim * 2**i, MAX_DIM)
        #         layers += [Conv2dBlock(
        #             n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn='tanh'
        #         )]
        #         n_in = n_out
        # self.enc_layers = nn.ModuleList(layers)
        
        layers = []
        self.dim_attrs = n_attrs * dim_per_attr  
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                # n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                # n_in = n_in + dim_attrs if self.inject_layers > i else n_in # inject attr
            else: # last layer
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='sigmoid'
                    # n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.dec_layers = nn.ModuleList(layers)
    
    def encode(self, x):
        z = x
        # zs = []
        for layer in self.enc_layers:
            z = layer(z)
            # zs.append(z)
        return z

    def decode(self, z, a, phase='normal'):
        if phase == 'erase':
            z = z.clone()
            a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, self.dim_per_attr, self.f_size, self.f_size)
            # print(z.size())
            # print(a_tile.size())
            z[:,:self.dim_attrs,:,:] = a_tile 
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
        return z
    
    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a, phase='erase')
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        if mode == 'dec_erase':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a, phase='erase')
        raise Exception('Unrecognized mode: ' + mode)
